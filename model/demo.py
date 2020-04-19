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

def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    tweet_data = TweetData(batch_size=4, file_size=100)
    label2tag = tweet_data.label_generator.label2tag
    text_vocab_size = tweet_data.label_generator.text_vocab.n_words
    label_vocab_size = tweet_data.label_generator.label_num

    cnn_rnn = CNN_RNN(text_vocab_size=text_vocab_size, text_embed_size=128, text_hidden_size = 128, \
            label_vocab_size=label_vocab_size, label_hidden_size = 128, resnet_version="resnet18", train_resnet=False)
    
    MODEL_PATH = "/home/feiyi/SI699_Hashtag/serialized/cross.pt"

    cnn_rnn.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    cnn_rnn.eval()

    for batch_data in tweet_data.dataloaders["train"]:
        text = batch_data["text"].to(device)
        image = batch_data["image"].to(device)
        label = batch_data["label"].to(device)
        text_length = batch_data["text_length"].to(device)

        with torch.no_grad():
            # Take the most probable predict from beam search
            predicts = cnn_rnn.sample(text, image, text_length, path_length=5, beam_width=10)[:,:,0]
        
        tag_pred = [[label2tag[int(x.item())] for x in predict] for predict in predicts]
        print("beam_search predicts: \t", tag_pred)

        with torch.no_grad():
            predicts = cnn_rnn.sample(text, image, text_length, path_length=5, beam_width=None)
        tag_pred = [[label2tag[int(x.item())] for x in predict] for predict in predicts]
        print("greedy predicts: \t", tag_pred)
        
        tag_gt = [[label2tag[int(x.item())] for x in l[1:]] for l in label]
        print("truth: \t \t \t", tag_gt)
        print()

demo()