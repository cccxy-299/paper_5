import os
import random

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm

class AE_dataset(Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.root = os.path.join(root, mode)
        self.list_video_indexs = os.listdir(self.root)
        self.transform = transform
        self.dataset_list = []
        self._init_dataset_list()

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        sample_name = self.dataset_list[idx]
        video_index = sample_name.split('_')[0]

        image_path = os.path.join(self.root, video_index, sample_name)

        image = Image.open(image_path).convert("RGB")  # 读取并转换为RGB格式
        tensor_image = self.transform(image)  # C H W
        return tensor_image

    def _init_dataset_list(self):
        for video_index in self.list_video_indexs:
            frames_path = os.path.join(self.root,video_index)
            list_frames = os.listdir(frames_path)

            for frame_name in list_frames:
                self.dataset_list.append(frame_name)



class SO_TAD(Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.appendix_path = os.path.join(root, "Appendix.txt")
        self.root = os.path.join(root, mode)
        self.key_frames = {}  # Array of sample video names
        self.list_video_indexs = os.listdir(self.root)
        self.transform = transform
        self.mode = mode
        self._init_label_dic()

    def __len__(self):
        return len(self.list_video_indexs)

    def __getitem__(self, idx):
        frame_number = []
        sample_name = self.list_video_indexs[idx]
        sample_path = os.path.join(self.root,sample_name)
        list_frames = os.listdir(sample_path)
        list_frames.sort()
        sample_label = 0.0
        label = [] # 0-no accident 1-have accident
        accident_frame = -1 # 事故帧
        accident_frame_index = -1
        if int(sample_name) < 400:
            sample_label = 1.0
            accident_frame = self.key_frames[int(sample_name)]
            for i in range(len(list_frames)):
                sequence_index = int(list_frames[i].split('.')[0][-6:])  # 获取后六位数字（"000024"）
                if sequence_index>=accident_frame:
                    label = [0] * i + [1] * (len(list_frames) - i)
                    accident_frame_index = i
                    break
        else:
            label = [0] * len(list_frames)
        sample = []
        for frame_name in list_frames:
            frame_number.append(int(frame_name.split('.')[0][-6:]))
            image_path = os.path.join(self.root, sample_name, frame_name)
            image = Image.open(image_path).convert("RGB")  # 读取并转换为RGB格式
            tensor_image = self.transform(image)  # C H W
            sample.append(tensor_image)
        tensor_sequence = torch.stack(sample)  # Sequ C H W
        if sample_label == 0 and tensor_sequence.shape[0]>60:
            rand_length = random.randint(25,60)
            tensor_sequence = tensor_sequence[:rand_length,:,:,:]
            label = label[:rand_length]
            frame_number = frame_number[:rand_length]
        return tensor_sequence, torch.Tensor(label), sample_label,accident_frame_index,frame_number,accident_frame,sample_name

    def _init_label_dic(self):
        """
            Get keyframes of accident samples
            by cxy
            Args:
                root: Path to Appendix.txt

            Returns:
                dic = {"1":36,"2":96}
            """
        # txt_path = os.path.join(self.root, "Appendix.txt")
        # print(len(self.list_video_indexs))
        with open(self.appendix_path, 'r') as f:
            line = f.readline()
            while line:
                video_name = int(line.split(".")[0])  # Sample video name
                key_frame = line.split("\t")[1].rstrip("\n").rstrip()  # Keyframes corresponding to the sample video
                self.key_frames[video_name] = int(key_frame)
                line = f.readline()

if __name__ == "__main__":
    # print(0)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    root = r'G:\dataset\paper5\so_tad'
    train_dataset = SO_TAD(root, mode='test',transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=4,
                              drop_last=False)
    label = []
    names = []
    for i, data in enumerate(tqdm(train_loader, leave=False)):
        label.append(data[2].item())
        names.append(int(data[6][0]))

    print(label, names)
        # print(f'{data[0].shape[1]}, {data[1].shape[1]}')
        # print(type(data[2]))