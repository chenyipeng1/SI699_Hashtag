import torch
from dataset import TweetDataset, LabelGenerator
import torch.nn as nn
from torchvision import models, transforms
import time
import os
import copy
import pickle


class TweetData():
    """
    Generate 
    1. self.dataloaders: Tweet Dataloaders for training, validation and testing usage.
    2. self.label_generator: Label Generator with three dictionaries tag2label, tag2freq, label2tag

    Input:
        batch_size: batch size of data
        csv_file: csv file with hashtag, text, image and image_path
        root_dir: directory of images
        data_transform: transform applied on images
        split_ratio: ratio of trainset, valset and testset
    
    """
    def __init__(self, batch_size, \
        csv_file="/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/OneMonthFilter846.csv", \
        root_dir="/scratch/si699w20_cbudak_class_root/si699w20_cbudak_class/shared_data/JI_team/data/dataset/OneMonthData/Image/10033", \
        data_transform= transforms.Compose([
                #transforms.ToPILImage(mode="RGB"),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]), \
        split_ratio=(0.7, 0.1, 0.2), file_size=None):

        assert(sum(split_ratio) == 1)
        self.label_generator = LabelGenerator(csv_file, file_size=file_size)
        dataset = TweetDataset(csv_file=csv_file, root_dir=root_dir, tag2label=self.label_generator.tag2label, \
                            text_vocab=self.label_generator.text_vocab, transform=data_transform, file_size=file_size)
        train_size = int(len(dataset) * split_ratio[0])
        val_size = int(len(dataset) * split_ratio[1])
        test_size = len(dataset) - train_size - val_size
        
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        self.datasets = {"train": train_set, "val": test_set, "test": val_set}
        self.dataset_sizes = {x: len(self.datasets[x]) for x in ["train", "val", "test"]}

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4, drop_last=True, collate_fn=dataset.collate_fn)
                    for x in ["train", "val", "test"]}
    def show_top_frequency(self, k=10):
        print("Top ", k, " hashtags with frequency")
        print(sorted(self.label_generator.tag2freq.items(), key=lambda x: x[1], reverse=True)[:k])    


if __name__ == "__main__":
    """
    To use dataloader, treat it as an generator
    """
    tweet_data = TweetData(batch_size=2, file_size=100)
    for batch_data in tweet_data.dataloaders["train"]:
        # print(batch_data)
        # print(batch_data['text'])
        break
