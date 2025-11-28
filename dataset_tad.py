import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm

class UCF(Dataset):
    def __init__(self, root,  transform=None):
        self.appendix_path = os.path.join(root, "vals.txt")
        self.root = os.path.join(root, "data")
        self.key_frames = {}  # Array of sample video names
        self.list_video_indexs = os.listdir(self.root)
        self.transform = transform
        self._init_label_dic()

    def __len__(self):
        return len(self.list_video_indexs)

    def __getitem__(self, idx):
        frame_number = []
        sample_name = self.list_video_indexs[idx]
        sample_path = os.path.join(self.root,sample_name)
        list_frames = os.listdir(sample_path)
        # list_frames = sorted(list_frames, key=lambda x: int(x.split('.')[0]))
        sample_label = 0.0

        label = [] # 0-no accident 1-have accident
        accident_frame = self.key_frames[sample_name]
        if accident_frame != -1 :
            sample_label = 1.0

        accident_frame_index = -1
        sample = []
        for i in range(len(list_frames)):
            frame_name = list_frames[i]
            if sample_label == 1.0:
                if int(frame_name.split('.')[0].split('_')[-1]) >= accident_frame:
                    accident_frame_index = i
            frame_number.append(int(frame_name.split('.')[0].split('_')[-1]))
            image_path = os.path.join(self.root, sample_name, frame_name)
            image = Image.open(image_path).convert("RGB")  # 读取并转换为RGB格式
            tensor_image = self.transform(image)  # C H W
            sample.append(tensor_image)
        tensor_sequence = torch.stack(sample)  # Sequ C H W
        return tensor_sequence, torch.Tensor([0.0]), sample_label,accident_frame_index,frame_number,accident_frame

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
        with open(self.appendix_path, 'r') as f:
            line = f.readline()
            while line:
                video_name = line.split(" ")[0]
                key_frame = line.split(" ")[2]
                video_name = video_name.split(".")[0]
                self.key_frames[video_name] = int(key_frame)
                line = f.readline()

class TAD(Dataset):
    def __init__(self, root,  transform=None):
        self.appendix_path = os.path.join(root, "vals.txt")
        self.root = os.path.join(root, "data")
        self.key_frames = {}  # Array of sample video names
        self.list_video_indexs = os.listdir(self.root)
        self.transform = transform
        self._init_label_dic()

    def __len__(self):
        return len(self.list_video_indexs)

    def __getitem__(self, idx):
        frame_number = []
        sample_name = self.list_video_indexs[idx]
        sample_path = os.path.join(self.root,sample_name)
        list_frames = os.listdir(sample_path)
        list_frames = sorted(list_frames, key=lambda x: int(x.split('.')[0]))
        sample_label = 0.0

        label = [] # 0-no accident 1-have accident
        accident_frame = self.key_frames[sample_name]
        if accident_frame != -1 :
            sample_label = 1.0

        accident_frame_index = -1
        sample = []
        for i in range(0,len(list_frames),6):
            frame_name = list_frames[i]
            if sample_label == 1.0:
                if int(frame_name.split('.')[0]) >= accident_frame:
                    accident_frame_index = i // 6
            frame_number.append(int(frame_name.split('.')[0]))
            image_path = os.path.join(self.root, sample_name, frame_name)
            image = Image.open(image_path).convert("RGB")  # 读取并转换为RGB格式
            tensor_image = self.transform(image)  # C H W
            sample.append(tensor_image)
        tensor_sequence = torch.stack(sample)  # Sequ C H W
        return tensor_sequence, torch.Tensor([0.0]), sample_label,accident_frame_index,frame_number,accident_frame

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
        with open(self.appendix_path, 'r') as f:
            line = f.readline()
            while line:
                video_name = line.split(" ")[0]
                key_frame = line.split(" ")[2]
                self.key_frames[video_name] = int(key_frame)
                line = f.readline()

if __name__ == "__main__":
    # print(0)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    root = r'G:\dataset\ucf_crimes'
    # train_dataset = UCF(root,transform=transform)
    train_dataset = TAD(root, mode='test',transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=8,
                              drop_last=False)
    label = []
    names = []
    for i, data in enumerate(tqdm(train_loader, leave=False)):
        label.append(data[2])
        names.append(data[6])
    print(label,names)
        # print(f'{data[0].shape[1]}, {data[1].shape[1]}')
        # print(type(data[2]))