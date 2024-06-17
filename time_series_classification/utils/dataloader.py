import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import skvideo
# skvideo.setFFmpegPath('D:/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin')
# import skvideo.io
import cv2


def read_images_from_txt(images_path, resolution=320):
    images = []

    for line in images_path:
        image_path = line.strip()  # 去除行末尾的换行符和空格
        cv_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        image = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # 使用OpenCV读取图像
        image = cv2.resize(image, (resolution, resolution))
        if image is not None:
            images.append(image)

    images = np.array(images)  # 将图像列表转换为NumPy数组

    images = np.transpose(images, [3, 0, 1, 2])

    return images


class LungWindowDataset(Dataset):
    def __init__(self, annotation_path="E:/Big_Datasets/medical_images/time_series/annotations", frame_num=128,
                 dataset_type='train', resolution=320):
        super().__init__()

        self.annotation_path = annotation_path
        self.frame_num = frame_num
        self.resolution = resolution

        self.files = os.listdir(self.annotation_path)

        if dataset_type == 'train':
            with open('utils/train.txt', 'r', encoding='utf8') as f:
                indexes = [int(item.strip()) for item in f.readlines()]
            self.data_set = [self.files[index] for index in indexes]

        elif dataset_type == 'val':
            with open('utils/val.txt', 'r', encoding='utf8') as f:
                indexes = [int(item.strip()) for item in f.readlines()]
            self.data_set = [self.files[index] for index in indexes]

        elif dataset_type == 'test':
            with open('utils/test.txt', 'r', encoding='utf8') as f:
                indexes = [int(item.strip()) for item in f.readlines()]
            self.data_set = [self.files[index] for index in indexes]

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        images_path = self.data_set[index]
        image_series_array = []
        with open(os.path.join(self.annotation_path, images_path), 'r', encoding='utf8') as f:
            images_names = [item.strip() for item in f.readlines()]
            frame_indexes = sorted(np.random.choice(len(images_names), self.frame_num, replace=False))
            images_names_list = [images_names[item] for item in frame_indexes]
            image_series_array = read_images_from_txt(images_names_list, resolution=self.resolution)
            if images_names_list[0].count('good') > 0:
                series_label = 1
            else:
                series_label = 0

            image_series_array = np.true_divide(image_series_array, 255.0)

        return image_series_array, series_label


def collate_fn(batch):
    videos = []
    labels = []
    for video, label in batch:
        videos.append(video)
        labels.append(label)
    videos = torch.from_numpy(np.array(videos)).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.LongTensor)

    return videos, labels


def get_dataloader(batch_size=1, frame_num=64, resolution=320, n_workers=4):
    trainset = LungWindowDataset(frame_num=200, dataset_type='train')
    testset = LungWindowDataset(frame_num=200, dataset_type='test')

    # --------------- 训练、测试、验证dataloader ----------------- #
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                             num_workers=n_workers)
    validloader = DataLoader(dataset=testset, batch_size=1,  collate_fn=collate_fn, shuffle=False,
                             num_workers=n_workers)

    return trainloader, validloader




if __name__ == '__main__':
    trainset = LungWindowDataset()

    trainloader, validloader = get_dataloader(batch_size=1)

    input_video = next(iter(validloader))[0].shape
    print(input_video)









