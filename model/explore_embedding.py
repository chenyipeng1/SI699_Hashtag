import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_rnn import CNN_RNN
from dataloader import TweetData
from torch.nn.utils.rnn import pack_padded_sequence
import time
from PIL import ImageFile
from focal_loss import FocalLoss
ImageFile.LOAD_TRUNCATED_IMAGES = True

def explore():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    tweet_data = TweetData(batch_size=8, file_size=None)
    for batch_data in tweet_data.dataloaders["train"]:
        print("Seed fixed")
        print("For whole 847-data with batch_size 8")
        print("Below should be 11., 12., 12.,  8.,  6.,  7.,  9.,  7.")
        print(batch_data["text_length"])
        break
    label2tag = tweet_data.label_generator.label2tag
    text_vocab_size = tweet_data.label_generator.text_vocab.n_words
    label_vocab_size = tweet_data.label_generator.label_num

    cnn_rnn = CNN_RNN(text_vocab_size=text_vocab_size, text_embed_size=128, text_hidden_size = 128, \
            label_vocab_size=label_vocab_size, label_hidden_size = 128, resnet_version="resnet18", train_resnet=False)
    
    MODEL_PATH = "/home/feiyi/SI699_Hashtag/serialized/cross.pt"
    cnn_rnn.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

    weight = cnn_rnn.decoder.embed.weight
    k = 10

    for i in range(10):
        distance = torch.mv(weight, weight[i])
        topk_labels = torch.topk(distance, k)[1]
        res = [label2tag[label.item()] for label in topk_labels]
        print("label2tag[i] Top 10 similar Tags: ")
        print(res)
        print()

explore()
