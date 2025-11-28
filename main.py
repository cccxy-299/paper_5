import argparse
import os
import shutil
import sys

import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score, roc_curve, auc
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchinfo import summary
from torchvision import datasets, transforms
from distutils.dir_util import copy_tree
import torch
import random
import datetime
import warnings

from dataset import AE_dataset,SO_TAD
from losses import SSIM_L1_Loss, td_loss

from models.fcmae_model import convnextv2_atto,convnextv2_femto,convnextv2_pico, convnextv2_nano,convnextv2_tiny,convnextv2_base,convnextv2_large
from td_model import FrameTransformer, TransformerClassifier, CAATR_VarLen
from utils import Logger, tensor_to_numpy
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_tad import TAD, UCF
from thop import profile

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def pretrain_ae(model,train_loader,test_loader,logdir,args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_ae)
    criterion = SSIM_L1_Loss(alpha=0.85)
    print('--------Pretrain AE Start------------')
    best_SSIM = 0
    for epoch in range(args.pretrain_epochs):
        model.train()
        total_losses = 0
        ssim_l1_losses = 0
        ConvNeXt_losses = 0
        for x in train_loader:
            x = x.to(args.device)
            optimizer.zero_grad()
            next_loss, x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            total_loss = loss + next_loss
            total_loss.backward()
            optimizer.step()

            total_losses += total_loss.item()
            ssim_l1_losses += loss.item()
            ConvNeXt_losses += next_loss.item()
        print(f"[{epoch + 1}/{args.pretrain_epochs} Train], "
              f"total_losses: {total_losses / len(train_loader):.4f}, "
              f"ssim_l1_losses: {ssim_l1_losses / len(train_loader):.4f}, "
              f"ConvNeXt_losses: {ConvNeXt_losses / len(train_loader):.4f}, ")
        '''eval_AE'''
        if epoch == args.pretrain_epochs - 1 or epoch % args.eval_epoch == 0:
            model.eval()
            psnr_list = []
            ssim_list = []
            with torch.no_grad():
                for x in test_loader:
                    x = x.to(args.device)
                    _, x_hat, _ = model(x)
                    for i in range(x_hat.size(0)):
                        out_np = tensor_to_numpy(x_hat[i])
                        gt_np = tensor_to_numpy(x[i])

                        psnr = peak_signal_noise_ratio(gt_np, out_np)# data_range=1.0
                        ssim = structural_similarity(gt_np, out_np, multichannel=True)# data_range=1.0

                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                avg_psnr = np.mean(psnr_list)
                avg_ssim = np.mean(ssim_list)
                # if avg_ssim > best_SSIM and args.is_save_log:
                #     best_SSIM = avg_ssim
                #     torch.save(model.state_dict(), os.path.join(logdir, 'ae_best.pth'))
                # if epoch == args.pretrain_epochs - 1 and args.is_save_log:
                #     torch.save(model.state_dict(), os.path.join(logdir, 'ae_final.pth'))
                print(f"[{epoch + 1}/{args.pretrain_epochs} Test], "
                      f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    print('--------Pretrain AE End------------')


def main(args):
    img_size = 128
    if args.seed is not None:

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # logging
    logdir = None
    if args.is_save_log:
        logdir = os.path.join(args.log_dir,f'{args.backbone}_{datetime.datetime.today():%m-%d_%H-%M-%S}')
        # logdir = f'G:\code\paper_5\\logs/' \
        #          f'{datetime.datetime.today():%Y-%m-%d_%H-%Mh3-%S}'
        os.makedirs(logdir, exist_ok=True)

        copy_tree('../tad', logdir + '/scripts')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )

    device = args.device
    print(f"device: {device}")
    print(args)

    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset_encoder_train = AE_dataset(args.dataset_root, "train", transform)
    dataset_encoder_test = AE_dataset(args.dataset_root, "test", transform)

    encoder_train_loader = DataLoader(dataset_encoder_train, batch_size=args.pretrain_batch_size, shuffle=True,
                                      num_workers=args.num_workers)
    encoder_test_loader = DataLoader(dataset_encoder_test, batch_size=args.pretrain_batch_size, shuffle=False,
                                     num_workers=args.num_workers)
    # torch.save(model_gat.state_dict(), os.path.join(logdir, 'teacher_gat_final.pth'))
    # use pico
    # convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano, convnextv2_tiny, convnextv2_base, convnextv2_large
    if args.backbone == 'convnextv2_atto':
        model_ae = convnextv2_atto(img_size=img_size)
        model_dim = 320
    elif args.backbone == 'convnextv2_femto':
        model_ae = convnextv2_femto(img_size=img_size)
        model_dim = 384
    elif args.backbone == 'convnextv2_pico':
        model_ae = convnextv2_pico(img_size=img_size)
        model_dim = 512
    elif args.backbone == 'convnextv2_nano':
        model_ae = convnextv2_nano(img_size=img_size)
        model_dim = 640
    elif args.backbone == 'convnextv2_tiny':
        model_ae = convnextv2_tiny(img_size=img_size)
        model_dim = 768
    elif args.backbone == 'convnextv2_base':
        model_ae = convnextv2_base(img_size=img_size)
        model_dim = 1024
    # elif args.backbone == 'convnextv2_large':
    #     model_ae = convnextv2_large()
    #     model_dim = 1536
    model_ae = model_ae.to(device)
    summary(model_ae, (1, 3, img_size, img_size))
    # inputs = torch.randn(1, 3, 224, 224)
    # inputs = inputs.to(device)
    # flops, params = profile(model_ae, inputs=(inputs,None,0.6,True))
    # print(f"{args.backbone} model_ae, FLOPs: {round(flops / 1e9, 4)} G, params:{round(params / 1e6, 4)} M")
    # del inputs

    pretrain_ae(model_ae,encoder_train_loader,encoder_test_loader,logdir,args)

    # 开始训练事故检测模型
    print("-------Model TD Staring--------")
    model_td = FrameTransformer(model_dim=model_dim)
    model_td = model_td.to(device)

    model_classification = TransformerClassifier()
    model_classification = model_classification.to(device)

    model_caatr = CAATR_VarLen(dim=model_dim)
    model_caatr = model_caatr.to(device)

    summary(model_td, (1, 30, model_dim))
    # summary(model_caatr, ((1, 30, model_dim),(1)))
    summary(model_classification, (1, 30, 1))

    # inputs = torch.randn(1, 30, model_dim)
    # inputs = inputs.to(device)
    # mask = torch.zeros((1, 30), device=inputs.device).bool()
    # flops, params = profile(model_td, inputs=(inputs,mask))
    # print(f"{args.backbone} model_td, FLOPs: {round(flops / 1e9, 4)} G, params:{round(params / 1e6, 4)} M")
    #
    # inputs = torch.randn(1, 30, 1)
    # inputs = inputs.to(device)
    # flops, params = profile(model_classification, inputs=(inputs,mask))
    # print(f"{args.backbone} model_classification, FLOPs: {round(flops / 1e9, 4)} G, params:{round(params / 1e6, 4)} M")

    # summary(model_td, (1, 3, 224, 224))
    optimizer = torch.optim.Adam([{'params':model_ae.encoder.parameters(),'params':model_td.parameters(),'params':model_caatr.parameters(),'params':model_classification.parameters()}], lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    train_dataset = SO_TAD(args.dataset_root,mode="train", transform=transform)
    test_dataset = SO_TAD(args.dataset_root, mode="test", transform=transform)

    td_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
    td_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)

    # tad_test_dataset = TAD(args.tad_root, transform=transform)
    tad_test_dataset = UCF(args.tad_root, transform=transform)
    tad_test_loader = DataLoader(tad_test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    best_f1 = 0
    for epoch in range(args.num_epochs):
        model_td.train()
        model_ae.train()
        model_classification.train()
        model_caatr.train()
        scaler = GradScaler()
        total_losses = []
        cos_losses = []
        frame_cls_losses = []
        sample_cls_losses = []
        all_pred = []
        all_label = []
        for x, label, sample_label,accident_index,frame_number,accident_frame in td_train_loader:
            x = x.to(device)
            label = label.to(device)
            accident_index = accident_index.to(device)
            sample_label = sample_label.to(device)

            optimizer.zero_grad()

            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            nums = T // args.pretrain_batch_size
            last = T % args.pretrain_batch_size
            frame_emb = []
            for i in range(nums):
                start = i*args.pretrain_batch_size
                end = (i+1)*args.pretrain_batch_size
                feature = model_ae(x[start:end,:,:,:], only_encoder=True)
                frame_emb.append(feature)
            if last !=0:
                feature = model_ae(x[-last:, :, :, :], only_encoder=True)
                frame_emb.append(feature)
            frame_emb = torch.cat(frame_emb,dim=0)  # Sequ 512
            frame_emb = frame_emb.unsqueeze(dim=0)

            T_target = 84 # 统一长度
            B, T_current, D = frame_emb.shape

            if T_current < T_target:
                pad_len = T_target - T_current
                pad = torch.zeros((B, pad_len, D), device=frame_emb.device)
                frame_emb = torch.cat([frame_emb, pad], dim=1)  #  frame_emb  [B, 84, 512]

                attention_mask = torch.cat([
                    torch.zeros((B, T_current), device=frame_emb.device),
                    torch.ones((B, pad_len), device=frame_emb.device)
                ], dim=1).bool()  # [B, 84]
            elif T_current > T_target:
                frame_emb = frame_emb[:, :T_target, :]
                attention_mask = torch.zeros((B, T_target), device=frame_emb.device).bool()
            else:
                attention_mask = torch.zeros((B, T_target), device=frame_emb.device).bool()
            # with torch.cuda.amp.autocast():
            frame_emb = model_caatr(frame_emb,[T_current])
            out = model_td(frame_emb, mask=attention_mask)  # out: [B, T, 1]

            B, T_current, D = out.shape
            if T_current < T_target:
                pad_len = T_target - T_current
                pad = torch.zeros((B, pad_len, D), device=out.device)
                out = torch.cat([out, pad], dim=1)  #  frame_emb  [B, 84, 512]
            elif T_current > T_target:
                out = out[:, :T_target, :]

            accident_index = accident_index[0].item()

            num_4_interval = 2
            cos_loss = 0
            if sample_label[0] == 1:
                if accident_index<4:
                    num_4_interval = 1
                T_sub_2 = frame_emb[:,accident_index - num_4_interval*2:accident_index - num_4_interval, :]
                T_sub_1 = frame_emb[:,accident_index - num_4_interval:accident_index, :]
                T_accident = frame_emb[:,accident_index:accident_index + num_4_interval, :]
                T_sub_2 = T_sub_2.view(1,-1)
                T_sub_1 = T_sub_1.view(1,-1)
                T_accident = T_accident.view(1,-1)
                cos_sim_1 = F.cosine_similarity(T_sub_2, T_sub_1)  # output shape: [T]
                cos_sim_2 = F.cosine_similarity(T_accident, T_sub_1)  # output shape: [T]
                cos_loss = F.relu(cos_sim_2 - cos_sim_1 + args.margin)
            else:
                key_index = random.randint(4, frame_emb.shape[1]-5)
                T_sub_2 = frame_emb[:,key_index - num_4_interval*2:key_index - num_4_interval, :]
                T_sub_1 = frame_emb[:,key_index - num_4_interval:key_index, :]
                T_sub_0 = frame_emb[:,key_index:key_index+num_4_interval, :]
                T_sub_2 = T_sub_2.view(1,-1)
                T_sub_1 = T_sub_1.view(1,-1)
                T_sub_0 = T_sub_0.view(1,-1)
                cos_sim_1 = F.cosine_similarity(T_sub_2, T_sub_1)  # output shape: [T]
                cos_sim_2 = F.cosine_similarity(T_sub_0, T_sub_1)  # output shape: [T]
                cos_loss = F.relu(args.lower_bound - (cos_sim_1-cos_sim_2)) + F.relu((cos_sim_1-cos_sim_2) - args.upper_bound)

            pos_weight = torch.tensor([29000 / (3800 + 1e-6)])

            pos_weight = pos_weight.to(device)
            frame_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(out.squeeze(-1)[:, :label.shape[1]], label)
            out_cls = model_classification(out,mask=attention_mask)
            pos_weight = torch.tensor([335 / (190 + 1e-6)])
            pos_weight = pos_weight.to(device)
            cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(out_cls.squeeze(-1), sample_label)

            total_loss = cls_loss*1 + cos_loss*1+ frame_loss * 0.5
            # total_loss = cls_loss * 1 + frame_loss * 0.5
                # total_loss = cos_loss * 1+ frame_loss*0.01
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_losses.append(total_loss.item())
            cos_losses.append(cos_loss.item())
            frame_cls_losses.append(frame_loss.item())
            sample_cls_losses.append(cls_loss.item())
            all_pred.extend(out_cls.detach().squeeze(-1).cpu().numpy())
            all_label.extend(sample_label.cpu().numpy())
        scheduler.step()
        print(f"[{epoch + 1}/{args.num_epochs}], "
              f"total_losses: {sum(total_losses) / len(total_losses):.4f}, "
              f"cos_losses: {sum(cos_losses) / len(cos_losses):.4f} "
              f"sample_cls_losses: {sum(sample_cls_losses) / len(sample_cls_losses):.4f}, "
              f"frame_cls_losses: {sum(frame_cls_losses) / len(frame_cls_losses):.4f}, "
              )

        if epoch == args.num_epochs - 1 or epoch % args.eval_epoch == 0:
            model_td.eval()
            model_ae.eval()
            model_classification.eval()
            model_caatr.eval()
            all_pred = []
            all_prob= []
            all_label = []
            frame_res = []
            tp_num = 0
            sum_dis = 0
            frame_number_list = []
            accident_frame_list=[]
            with torch.no_grad():
                for x, _, sample_label, _,frame_number,accident_frame in td_test_loader:
                    frame_number_list.append(frame_number)
                    accident_frame_list.append(accident_frame)
                    x = x.to(device)
                    # label = label.to(device)
                    # accident_index = accident_index.to(device)
                    sample_label = sample_label.to(device)

                    B, T, C, H, W = x.shape
                    x = x.view(B * T, C, H, W)
                    nums = T // args.pretrain_batch_size
                    last = T % args.pretrain_batch_size
                    frame_emb = []
                    for i in range(nums):
                        start = i * args.pretrain_batch_size
                        end = (i + 1) * args.pretrain_batch_size
                        feature = model_ae(x[start:end, :, :, :], only_encoder=True)
                        frame_emb.append(feature)
                    if last!=0:
                        feature = model_ae(x[-last:, :, :, :], only_encoder=True)
                        frame_emb.append(feature)
                    frame_emb = torch.cat(frame_emb, dim=0)  # Sequ 512
                    frame_emb = frame_emb.unsqueeze(dim=0)

                    T_target = 84
                    B, T_current, D = frame_emb.shape

                    # if T_current < T_target:
                    #     pad_len = T_target - T_current
                    #     pad = torch.zeros((B, pad_len, D), device=frame_emb.device)
                    #     frame_emb = torch.cat([frame_emb, pad], dim=1)  # 现在 frame_emb 才是 [B, 84, 512]
                    #
                    #     attention_mask = torch.cat([
                    #         torch.zeros((B, T_current), device=frame_emb.device),
                    #         torch.ones((B, pad_len), device=frame_emb.device)
                    #     ], dim=1).bool()  # [B, 84]
                    # elif T_current > T_target:
                    #     frame_emb = frame_emb[:, :T_target, :]
                    #     attention_mask = torch.zeros((B, T_target), device=frame_emb.device).bool()
                    # else:
                    #     attention_mask = torch.zeros((B, T_target), device=frame_emb.device).bool()
                    frame_emb = model_caatr(frame_emb, [T_current])
                    # out = model_td(frame_emb,mask=attention_mask)  # out: [1, T, 1]
                    out = model_td(frame_emb)  # out: [1, T, 1]
                    frame_res.append(F.sigmoid(out).detach().squeeze(-1).cpu().numpy())  # 获取预测结果
                    # B, T_current, D = out.shape
                    # if T_current < T_target:
                    #     pad_len = T_target - T_current
                    #     pad = torch.zeros((B, pad_len, D), device=out.device)
                    #     out = torch.cat([out, pad], dim=1)  # 现在 frame_emb 才是 [B, 84, 512]
                    # elif T_current > T_target:
                    #     out = out[:, :T_target, :]

                    # res_out = F.sigmoid(out).detach().squeeze(-1).cpu().numpy()


                    out_cls = model_classification(out)
                    all_prob.extend(out_cls.detach().squeeze(-1).cpu().numpy())
                    out_cls = F.sigmoid(out_cls)
                    all_pred.extend(out_cls.detach().squeeze(-1).cpu().numpy())
                    all_label.extend(sample_label.cpu().numpy())

                auc_score = roc_auc_score(all_label, all_pred)
                fpr, tpr, thresholds = roc_curve(all_label, all_pred)
                youden_j = tpr - fpr
                best_threshold = thresholds[youden_j.argmax()]
                binary = [int((x > best_threshold)) for x in all_pred]

                for index in range(len(binary)):
                    if all_label[index]==1:
                        tp_num = tp_num + 1
                        if binary[index]==0:
                            sum_dis += np.square(250)
                        else: # 3
                            tmp_frame_number = frame_number_list[index]
                            tmp_key_frame = accident_frame_list[index]
                            tmp_pred_index = frame_res[index].argmax()
                            # if tmp_pred_index>= len(tmp_frame_number):
                            #     a = 1
                            #     print(a)
                            pred_frame = tmp_frame_number[tmp_pred_index]
                            if pred_frame<tmp_key_frame:
                                dis = 250
                            else:
                                dis = abs(pred_frame - tmp_key_frame)
                            sum_dis += np.square(dis)
                nrmse = (min(np.sqrt(sum_dis / tp_num), 250)) / 250

                epoch_acc = accuracy_score(all_label, binary)
                epoch_precision = precision_score(all_label, binary, average='weighted', zero_division=0)
                epoch_recall = recall_score(all_label, binary, average='weighted', zero_division=0)
                epoch_f1 = f1_score(all_label, binary, average='weighted', zero_division=0)
                recall_1 = classification_report(all_label, binary, output_dict=True)["1.0"]["recall"]
                recall_0 = classification_report(all_label, binary, output_dict=True)["0.0"]["recall"]
                best_threshold = float(best_threshold)
                nrmse = float(nrmse)
                print(f"[Dataset SO-TAD] threshold:{best_threshold:.2f} "
                      f'NRMSE: {nrmse:.4f} '
                      f'Acc: {epoch_acc:.4f} '
                      f'Precision: {epoch_precision:.4f} '
                      f'Recall: {epoch_recall:.4f} '
                      f'F1: {epoch_f1:.4f} '
                      f'Recal_0: {recall_0:.4f} '
                      f'Recal_1: {recall_1:.4f} '
                      f'AUC: {auc_score:.4f} ')
                if epoch_f1 > best_f1 and args.is_save_log:
                    best_f1 = epoch_f1
                    torch.save(model_ae.state_dict(), os.path.join(logdir, 'ae_best.pth'))
                    torch.save(model_td.state_dict(), os.path.join(logdir, 'td_best.pth'))
                    torch.save(model_classification.state_dict(), os.path.join(logdir, 'cls_best.pth'))
                    torch.save(model_caatr.state_dict(), os.path.join(logdir, 'caatr_best.pth'))
                if epoch == args.num_epochs - 1 and args.is_save_log:
                    torch.save(model_ae.state_dict(), os.path.join(logdir, 'ae_final.pth'))
                    torch.save(model_td.state_dict(), os.path.join(logdir, 'td_final.pth'))
                    torch.save(model_classification.state_dict(), os.path.join(logdir, 'cls_final.pth'))
                    torch.save(model_caatr.state_dict(), os.path.join(logdir, 'caatr_final.pth'))
                    print(f'all_label:{all_label}')
                    print(f'all_pred:{all_pred}')

            '''TAD Test'''
            all_pred = []
            all_label = []
            all_prob = []
            frame_res = []
            tp_num = 0
            sum_dis = 0
            frame_number_list = []
            accident_frame_list = []
            with torch.no_grad():
                for x, _, sample_label, _, frame_number, accident_frame in tad_test_loader:
                    frame_number_list.append(frame_number)
                    accident_frame_list.append(accident_frame)
                    x = x.to(device)
                    # label = label.to(device)
                    # accident_index = accident_index.to(device)
                    sample_label = sample_label.to(device)

                    B, T, C, H, W = x.shape
                    x = x.view(B * T, C, H, W)
                    nums = T // args.pretrain_batch_size
                    last = T % args.pretrain_batch_size
                    frame_emb = []
                    for i in range(nums):
                        start = i * args.pretrain_batch_size
                        end = (i + 1) * args.pretrain_batch_size
                        feature = model_ae(x[start:end, :, :, :], only_encoder=True)
                        frame_emb.append(feature)
                    if last != 0:
                        feature = model_ae(x[-last:, :, :, :], only_encoder=True)
                        frame_emb.append(feature)
                    frame_emb = torch.cat(frame_emb, dim=0)  # Sequ 512
                    frame_emb = frame_emb.unsqueeze(dim=0)

                    # 标准化序列长度为 84
                    T_target = 84
                    B, T_current, D = frame_emb.shape

                    # if T_current < T_target:
                    #     pad_len = T_target - T_current
                    #     pad = torch.zeros((B, pad_len, D), device=frame_emb.device)
                    #     frame_emb = torch.cat([frame_emb, pad], dim=1)  # 现在 frame_emb 才是 [B, 84, 512]
                    #
                    #     attention_mask = torch.cat([
                    #         torch.zeros((B, T_current), device=frame_emb.device),
                    #         torch.ones((B, pad_len), device=frame_emb.device)
                    #     ], dim=1).bool()  # [B, 84]
                    # elif T_current > T_target:
                    #     frame_emb = frame_emb[:, :T_target, :]
                    #     attention_mask = torch.zeros((B, T_target), device=frame_emb.device).bool()
                    # else:
                    #     attention_mask = torch.zeros((B, T_target), device=frame_emb.device).bool()

                    frame_emb = model_caatr(frame_emb, [T_current])
                    out = model_td(frame_emb)  # out: [1, T, 1]
                    # out = model_td(frame_emb,mask=attention_mask)  # out: [1, T, 1]
                    frame_res.append(F.sigmoid(out).detach().squeeze(-1).cpu().numpy())  # 获取预测结果
                    # B, T_current, D = out.shape
                    # if T_current < T_target:
                    #     pad_len = T_target - T_current
                    #     pad = torch.zeros((B, pad_len, D), device=out.device)
                    #     out = torch.cat([out, pad], dim=1)  # 现在 frame_emb 才是 [B, 84, 512]
                    # elif T_current > T_target:
                    #     out = out[:, :T_target, :]

                    # res_out = F.sigmoid(out).detach().squeeze(-1).cpu().numpy()


                    # out_cls = model_classification(out,mask=attention_mask)
                    out_cls = model_classification(out)
                    all_prob.extend(out_cls.detach().squeeze(-1).cpu().numpy())
                    out_cls = F.sigmoid(out_cls)
                    all_pred.extend(out_cls.detach().squeeze(-1).cpu().numpy())
                    all_label.extend(sample_label.cpu().numpy())


                auc_score = roc_auc_score(all_label, all_pred)
                fpr, tpr, thresholds = roc_curve(all_label, all_pred)
                youden_j = tpr - fpr
                best_threshold = thresholds[youden_j.argmax()]
                binary = [int((x > best_threshold)) for x in all_pred]

                for index in range(len(binary)):
                    if all_label[index] == 1:
                        tp_num = tp_num + 1
                        if binary[index] == 0:
                            sum_dis += np.square(250)
                        else:  # 3
                            tmp_frame_number = frame_number_list[index]
                            tmp_key_frame = accident_frame_list[index]
                            tmp_pred_index = frame_res[index].argmax()
                            # if tmp_pred_index>= len(tmp_frame_number):
                            #     a = 1
                            #     print(a)
                            pred_frame = tmp_frame_number[tmp_pred_index]
                            if pred_frame < tmp_key_frame:
                                dis = 250
                            else:
                                dis = abs(pred_frame - tmp_key_frame)
                            sum_dis += np.square(dis)
                nrmse = (min(np.sqrt(sum_dis / tp_num), 250)) / 250

                epoch_acc = accuracy_score(all_label, binary)
                epoch_precision = precision_score(all_label, binary, average='weighted', zero_division=0)
                epoch_recall = recall_score(all_label, binary, average='weighted', zero_division=0)
                epoch_f1 = f1_score(all_label, binary, average='weighted', zero_division=0)
                recall_1 = classification_report(all_label, binary, output_dict=True)["1.0"]["recall"]
                recall_0 = classification_report(all_label, binary, output_dict=True)["0.0"]["recall"]

                best_threshold = float(best_threshold)
                nrmse = float(nrmse)
                print(f"[Dataset UCF] threshold:{best_threshold:.2f} "
                      f'NRMSE: {nrmse:.4f} '
                      f'Acc: {epoch_acc:.4f} '
                      f'Precision: {epoch_precision:.4f} '
                      f'Recall: {epoch_recall:.4f} '
                      f'F1: {epoch_f1:.4f} '
                      f'Recal_0: {recall_0:.4f} '
                      f'Recal_1: {recall_1:.4f} '
                      f'AUC: {auc_score:.4f} ')
                if epoch == args.num_epochs - 1 and args.is_save_log:
                    print(f'UCF all_label:{all_label}')
                    print(f'UCF all_pred:{all_pred}')
                    # print(f'UCF all_prob:{all_prob}')


if __name__ == "__main__":
    random_seed = int.from_bytes(os.urandom(4), byteorder='big')
    parser = argparse.ArgumentParser(description='No Name')
    parser.add_argument('--log_dir', type=str, default=r'G:\code\paper_5\logs')

    parser.add_argument('--dataset_root', type=str, default=r'G:\dataset\paper5\so_tad')
    # parser.add_argument('--tad_root', type=str, default=r'G:\dataset\TAD')
    parser.add_argument('--tad_root', type=str, default=r'G:\dataset\ucf_crimes')

    # parser.add_argument('--dataset_root', type=str, default=r'G:\dataset\paper5\min_test\so_tad')
    # parser.add_argument('--tad_root', type=str, default=r'G:\dataset\paper5\min_test\tad')
    # convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano, convnextv2_tiny, convnextv2_base
    parser.add_argument('--backbone', type=str, default='convnextv2_base')
    parser.add_argument('--seed', type=int, default=random_seed, help='random seed')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.set_defaults(is_save_log=True)
    parser.add_argument('--eval_epoch', type=int, default="1")


    parser.add_argument('--num_workers', type=int, default="4")
    # pre-train
    parser.add_argument('--pretrain_epochs', type=int, default="3")
    parser.add_argument('--pretrain_batch_size', type=int, default="16")
    #train
    parser.add_argument('--num_epochs', type=int, default="6")
    parser.add_argument('--batch_size', type=int, default="1")
    # cos loss Hyperparameter
    parser.add_argument('--margin', type=float, help='accident frame cos_sim margin', default="0.3")
    parser.add_argument('--lower_bound', type=float, help='no accident lower bound' ,default="0.2")
    parser.add_argument('--upper_bound', type=float, help='no accident upper bound',default="0.2")
    # learning rate
    parser.add_argument('--lr_ae', type=float, default="1e-4")
    parser.add_argument('--lr', type=float, default="1e-4")


    args = parser.parse_args()
    main(args)