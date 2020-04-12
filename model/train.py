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

def count_corrects(labels, predicts, label_lengths, k=None):
    corrects = 0
    batch_size, time_step = labels.shape
    if not k:
        for i in range(batch_size):
            corrects += torch.sum(labels[i,:label_lengths[i]] == predicts[i,:label_lengths[i]])
    else:
        # labels B, T
        # predicts B, T, k
        # label_lengths B
        for i in range(batch_size):
            label = labels[i,:label_lengths[i]]
            predict = predicts[i,:label_lengths[i]]
            label = label.unsqueeze(1).expand(label_lengths[i], k)
            correct, _ = (label == predict).long().max(dim=1)
            corrects += torch.sum(correct)    
    return corrects

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    file_size = 100
    tweet_data = TweetData(batch_size=4, file_size=file_size)
    text_vocab_size = tweet_data.label_generator.text_vocab.n_words
    label_vocab_size = tweet_data.label_generator.label_num

    cnn_rnn = CNN_RNN(text_vocab_size=text_vocab_size, text_embed_size=128, text_hidden_size = 512, \
            label_vocab_size=label_vocab_size, label_hidden_size = 256, resnet_version="resnet18", train_resnet=False)
    
    USE_FOCAL_LOSS = True
    SAVE_MODEL = True # save model every 5 epoches
    MODEL_PATH = "/home/feiyi/SI699_Hashtag/serialized/{}.pt".format(file_size) 

    cnn_rnn = cnn_rnn.to(device)
    epoch_num = 20
    k = 5 # Top k accuracy
    optimizer = torch.optim.Adam(cnn_rnn.parameters(), lr=0.001)

    criterion = None
    if USE_FOCAL_LOSS:
        print("Using Focal Loss")
        criterion = FocalLoss(gamma=2,ignore_index=0)
    else:
        print("Using Cross Entropy")   
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    

    for epoch in range(epoch_num):
        
        print('Epoch {}/{}'.format(epoch+1, epoch_num))
        print('-' * 10)
        
        for phase in ["train", "val"]:
            since = time.time()
            if phase == "train":
                cnn_rnn.train()
            else:
                cnn_rnn.eval()

            running_loss = 0.0
            running_corrects = 0
            running_corrects_topk = 0
            running_size = 0.0
            
            for batch_data in tweet_data.dataloaders[phase]:
                text = batch_data["text"].to(device)
                image = batch_data["image"].to(device)
                label = batch_data["label"].to(device)
                label_lengths = batch_data["label_length"].to(device)
                text_lengths = batch_data["text_length"].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = cnn_rnn.forward(text, image, label, text_lengths) # B, T, label_vocab_size
                    
                    label_trim = label[:,1:]
                    loss = criterion(outputs.transpose(2,1), label_trim)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                with torch.no_grad():
                    _, predicts = torch.max(outputs, dim=2) # B, T
                    _, predicts_topk = torch.topk(outputs, k=k, dim=2, sorted=False)
                    running_loss += loss.item() * torch.sum(label_lengths-1)
                    running_corrects += count_corrects(label_trim, predicts, label_lengths.long()-1)
                    running_corrects_topk += count_corrects(label_trim, predicts_topk, label_lengths.long()-1, k=k)
                    running_size += torch.sum(label_lengths-1)

            epoch_loss = running_loss / running_size
            epoch_acc = running_corrects / running_size
            epoch_acc_topk = running_corrects_topk / running_size
            time_elapsed = time.time() - since
            print('{} Loss:\t{:.4f} Top1 Acc: {:.4f} Top{} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(phase, epoch_loss, \
                epoch_acc, k, epoch_acc_topk, time_elapsed//60, time_elapsed%60))
        
        if SAVE_MODEL and (epoch + 1) % 5 == 0:
            torch.save(cnn_rnn.state_dict(), MODEL_PATH)

train()
