import os
import pandas as pd
import torch, torchvision
import collections
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader # need pip install pillow==6.1
import collections
from read_text import TextVocabulary

"""
def PopularSetGenerator(csv_file, limit=300):
    tag2num = collections.defaultdict(int)
    df = pd.read_csv(csv_file, lineterminator='\n', quotechar='"')
        #labels = []
    for idx in range(0, df.shape[0]):
        tag = df.loc[idx, "hashtags"][1:-1].split(",")[0]
        tag2num[tag] += 1
    sorted_items = sorted(tag2num.items(), reverse=True, key=lambda x: x[1])[0:limit]
    print(sum([item[1] for item in sorted_items]), " tweets used")
    return set([item[0] for item in sorted_items])
"""

class LabelGenerator():
    """
        Generate three dictionaries
        1. hashtag to label
        2. hashtag to frequency
        3. label to hashtag
    """
    def __init__(self, csv_file, popular_tags=None, file_size=None):
        self.tag2label = {} #{"<end>": 0, "<start>": 1}
        self.tag2freq = collections.defaultdict(int)
        self.label2tag = {} #{0: "<end>", 1: "<start>"}
        self.label_num = 0 #2
        self.text_vocab = TextVocabulary('tweet')
        df = pd.read_csv(csv_file, lineterminator='\n', quotechar='"')
        if file_size:
            df = df.iloc[0:file_size]
        print(df.shape[0], " tweets")
        for idx in range(0, df.shape[0]):
            tags = [x.strip() for x in df.loc[idx, "hashtags"][1:-1].split(",")][0]
            tags = [tags]
            for tag in tags:
                if popular_tags and tag not in popular_tags:
                    continue
                if tag not in self.tag2label:
                    self.tag2label[tag] = self.label_num
                    self.label2tag[self.label_num] = tag
                    self.label_num += 1
                self.tag2freq[tag] += 1

            text = df.loc[idx, "text"]
            text_preprocessed = self.text_vocab.preprocess(text)
            self.text_vocab.addSentence(text_preprocessed)

        print(self.label_num, " labels")

class TweetDataset(Dataset):
    """
    Tweet Dataset
    Each item is a dictionary
    {"text": text, "image": image, "label": label_list}
    """
    def __init__(self, csv_file, root_dir, tag2label, text_vocab, transform=None, max_text_len=50, max_label_len=10, file_size=None):
        self.df = pd.read_csv(csv_file, lineterminator='\n', quotechar='"')
        # for test usag
        if file_size:
            self.df = self.df.iloc[0:file_size]
        self.root_dir = root_dir
        self.transform = transform
        self.tag2label = tag2label
        self.text_vocab = text_vocab
        self.max_text_len = max_text_len
        self.max_label_len = max_label_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.df.loc[idx, "text"]
        img_name = os.path.join(self.root_dir, self.df.loc[idx, "path"])
        # tags = ["<start>"]
        tags = self.df.loc[idx, "hashtags"][1:-1].split(",")[0]
        image = io.imread(img_name)
        image = Image.fromarray(image).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text_preprocessed = self.text_vocab.preprocess(text)
        #return self.text_vocab.tensorFromSentence(text_preprocessed), image, torch.tensor([self.tag2label[x.strip()] for x in tags])
        return {"text": text, "image": image, "label": torch.tensor(self.tag2label[tags])}
        
    
    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples (image, caption).
        
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.
        Args:
            data: list of tuple (image, caption). 
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.
        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        # Sort a data list by caption length (descending order).
        
        data.sort(key=lambda x: len(x[2]), reverse=True)
        texts, images, labels = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)
        ### generate random data
        # images = torch.rand(images.shape) - 0.5
        
        # Merge labels (from tuple of 1D tensor to 2D tensor).
        label_lengths = [min(self.max_label_len, len(label)) for label in labels]
        label_stacked = torch.zeros(len(labels), max(label_lengths)).long()
        for i, label in enumerate(labels):
            end = label_lengths[i]
            label_stacked[i, :end] = label[:end]

        text_lengths = [min(self.max_text_len, len(text)) for text in texts]
        text_stacked = torch.zeros(len(texts), max(text_lengths)).long()
        for i, text in enumerate(texts):
            end = text_lengths[i]
            text_stacked[i, :end] = text[:end]
        
        ### generate random data
        # text_stacked = torch.zeros(text_stacked.shape).long()
        return {"text": text_stacked, "image": images, "label": label_stacked, \
            "label_length": torch.Tensor(label_lengths), "text_length": torch.Tensor(text_lengths)}

    def __len__(self):
        return self.df.shape[0]





