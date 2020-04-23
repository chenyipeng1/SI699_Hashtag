import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_rnn import CNN_RNN
from text_encoder import EncoderGRU
from dataloader import TweetData
from torch.nn.utils.rnn import pack_padded_sequence
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def count_corrects(label, predict, label_lengths):
    corrects = 0
    batch_size, time_step = label.shape
    for i in range(batch_size):
        corrects += torch.sum(label[i,:label_lengths[i]] == predict[i,:label_lengths[i]])
    return corrects

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    tweet_data = TweetData(batch_size=8, file_size=None)
    for batch_data in tweet_data.dataloaders["train"]:
        print(batch_data["text_length"])
        break
    text_vocab_size = tweet_data.label_generator.text_vocab.n_words
    label_vocab_size = tweet_data.label_generator.label_num
    text_rnn=EncoderGRU(input_size=text_vocab_size, embed_size=128, hidden_size=128,num_output=label_vocab_size).to(device)
    epoch_num = 20
    optimizer = torch.optim.Adam(text_rnn.parameters(), lr=0.001)
    #criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    MODEL_PATH = "/home/feiyi/SI699_Hashtag/serialized/text_basline.pt"
    for epoch in range(epoch_num):
        print('Epoch {}/{}'.format(epoch+1, epoch_num))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        running_size = 0.0

        for phase in ["train", "val", "test"]:
            since = time.time()
            for batch_data in tweet_data.dataloaders[phase]:
                #print(batch_data)
                #break
                text = batch_data["text"].to(device)
                # image = batch_data["image"].to(device)
                label = batch_data["label"].to(device)
                #label_lengths = batch_data["label_length"].to(device)
                text_lengths = batch_data["text_length"].to(device)
                optimizer.zero_grad()

                # outputs = cnn_rnn.forward(text, image, label, text_lengths) # B, T, label_vocab_size
                # _, predicts = torch.max(outputs, 2) # B, T
                # label_trim = label[:,1:]
                with torch.set_grad_enabled(phase == "train"):
                    outputs=text_rnn.forward(text,text_lengths)
                    index_nonzero = label.nonzero()
                    predicts = (outputs[(index_nonzero[:,0], index_nonzero[:,1])] > 0.5).long()
                    loss = criterion(outputs, label)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * index_nonzero.shape[0]
                running_corrects += torch.sum(predicts)
                running_size += index_nonzero.shape[0]

            epoch_loss = running_loss / running_size
            epoch_acc = running_corrects / running_size

            time_elapsed = time.time() - since
            print('{} Loss: {:.4f} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(phase, epoch_loss, epoch_acc, time_elapsed//60, time_elapsed%60))
        scheduler.step()
        torch.save(text_rnn.state_dict(), MODEL_PATH)
train()
