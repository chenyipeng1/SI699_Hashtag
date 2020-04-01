import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_rnn import CNN_RNN
from dataloader import TweetData
from torch.nn.utils.rnn import pack_padded_sequence

def count_corrects(label, predict, label_lengths):
    corrects = 0
    batch_size, time_step = label.shape
    for i in range(batch_size):
        corrects += torch.sum(label[i,:label_lengths[i]] == predict[i,:label_lengths[i]])
    return corrects

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    tweet_data = TweetData(batch_size=4)
    text_vocab_size = tweet_data.label_generator.text_vocab.n_words
    label_vocab_size = tweet_data.label_generator.label_num
    cnn_rnn = CNN_RNN(text_vocab_size=text_vocab_size, text_embed_size=30, text_hidden_size = 512, \
            label_vocab_size=label_vocab_size, label_embed_size = 512, label_hidden_size = 512)
    epoch_num = 20
    optimizer = torch.optim.Adam(cnn_rnn.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epoch_num):
        print('Epoch {}/{}'.format(epoch+1, epoch_num))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0

        for batch_data in tweet_data.dataloaders["train"]:
            text = batch_data["text"]
            image = batch_data["image"]
            label = batch_data["label"]
            label_lengths = batch_data["label_length"]
            optimizer.zero_grad()

            outputs = cnn_rnn.forward(batch_data) # B, T, label_vocab_size
            _, predicts = torch.max(outputs, 2) # B, T
            label_trim = label[:,1:]

            
            loss = criterion(outputs.transpose(2,1), label_trim)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * torch.sum(label_lengths-1)
            running_corrects += count_corrects(label_trim, predicts, label_lengths.long()-1)
            running_size += torch.sum(label_lengths-1)

        epoch_loss = running_loss / running_size
        epoch_acc = running_corrects / running_size
        phase = "train"
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

train()