import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from dataloader import TweetData
from model_attention import EncoderCNN, DecoderRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
	tweet_data = TweetData(batch_size=2)
	encoder = EncoderCNN().to(device)
	decoder = DecoderRNN(embed_size=16, hidden_size=16, vocab_size=10029, num_layers=1).to(device)
	for batch_data in tweet_data.dataloaders["train"]:
		images = batch_data["image"]
		labels = batch_data["label"]
		#print(labels)
		#print(labels[0])
		features = encoder(images)
		outputs = decoder(features, labels)
