import torch
import torch.nn as nn
import torch.nn.functional as F
from text_encoder import EncoderRNN
from model_attention import EncoderCNN, DecoderRNN
from dataloader import TweetData

class CNN_RNN(nn.Module):
    def __init__(self, text_vocab_size, text_embed_size, text_hidden_size, \
        label_vocab_size, label_hidden_size, resnet_version="resnet18", train_resnet=True):
        super(CNN_RNN, self).__init__()
        self.text_encoder = EncoderRNN(input_size=text_vocab_size, embed_size=text_embed_size, hidden_size=text_hidden_size)
        self.image_encoder = EncoderCNN(resnet_version=resnet_version, train_resnet=train_resnet)
        image_hidden_size = self.image_encoder.image_hidden_size
        self.decoder = DecoderRNN(vocab_size=label_vocab_size, image_hidden_size=image_hidden_size, \
            text_hidden_size=text_hidden_size, label_hidden_size=label_hidden_size, num_layers=1)

    def forward(self, text, image, label, text_length):

        text_features = self.text_encoder(text, text_length)
        #print("text feat: ", text_features.shape)
        image_features = self.image_encoder(image)
        #print("image feat: ", image_features.shape)
        predicts = self.decoder(text_features, image_features, label)
        #print("predicts: ", predicts.shape)
        return predicts
