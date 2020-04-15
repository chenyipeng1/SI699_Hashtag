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

        image_features = self.image_encoder(image)

        predicts = self.decoder(text_features, image_features, label)

        return predicts
    
    def sample_greedy(self, text, image, text_length, path_length=10):
        text_features = self.text_encoder(text, text_length)

        image_features = self.image_encoder(image)

        predicts = self.decoder.sample_greedy(text_features, image_features, path_length=path_length)

        return predicts
    
    #def sample_beam_search()