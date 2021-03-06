import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_rnn import CNN_RNN
from dataloader import TweetData
from torch.nn.utils.rnn import pack_padded_sequence
import time
from PIL import ImageFile
from focal_loss import FocalLoss
from prediction_analysis import PredictionAnalysis

ImageFile.LOAD_TRUNCATED_IMAGES = True

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    file_size = None
    tweet_data = TweetData(batch_size=8, file_size=file_size)
    text_vocab_size = tweet_data.label_generator.text_vocab.n_words
    label_vocab_size = tweet_data.label_generator.label_num
    for batch_data in tweet_data.dataloaders["train"]:
        print("Seed fixed")
        print("For whole 847-data with batch_size 8")
        print("Below should be 11., 12., 12.,  8.,  6.,  7.,  9.,  7.")
        print(batch_data["text_length"])
        break
    cnn_rnn = CNN_RNN(text_vocab_size=text_vocab_size, text_embed_size=128, text_hidden_size = 128, \
            label_vocab_size=label_vocab_size, label_hidden_size = 128, resnet_version="resnet18", train_resnet=False)
    
    SHOW_TOP_PREDICT_FREQ = True # Print top 10 predicted & groudtruth hashtags
    USE_FOCAL_LOSS = True
    SAVE_MODEL = True # save model every 5 epoches
    """
    !!! Change Path !!!
    """
    MODEL_PATH = "/home/feiyi/SI699_Hashtag/serialized/{}.pt".format(file_size)

    cnn_rnn = cnn_rnn.to(device)
    epoch_num = 20
    optimizer = torch.optim.Adam(cnn_rnn.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    k = 5 # Top k accuracy

    criterion = None
    if USE_FOCAL_LOSS:
        print("Using Focal Loss")
        criterion = FocalLoss(gamma=2, ignore_index=0, size_average=False)
    else:
        print("Using Cross Entropy")   
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    

    for epoch in range(epoch_num):
        
        print('Epoch {}/{}'.format(epoch+1, epoch_num))
        print('-' * 10)
        
        
        for phase in ["train", "val", "test"]:
            since = time.time()
            if phase == "train":
                cnn_rnn.train()
            else:
                cnn_rnn.eval()

            prediction_analysis = PredictionAnalysis(tweet_data.label_generator.label2tag)
            running_loss = 0.0
            running_corrects = 0
            running_corrects_topk = 0
            running_size = 0.0
            
            for batch_data in tweet_data.dataloaders[phase]:
                text = batch_data["text"].to(device)
                image = batch_data["image"].to(device)
                label = batch_data["label"].to(device)
                label_trim = label[:,1:]
                label_lengths = batch_data["label_length"].to(device)
                text_lengths = batch_data["text_length"].to(device)
                optimizer.zero_grad()

                if phase in ["train", "val"]:
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = cnn_rnn.forward(text, image, label, text_lengths) # B, T, label_vocab_size
                        
                        loss = criterion(outputs.transpose(2,1), label_trim)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    
                    with torch.no_grad():
                        _, predicts = torch.max(outputs, dim=2) # B, T
                        _, predicts_topk = torch.topk(outputs, k=k, dim=2, sorted=False) # B, T, k
                        running_loss += loss.item() * torch.sum(label_lengths-1)
                        running_corrects += prediction_analysis.count_corrects(label_trim, predicts, label_lengths.long()-1)
                        running_corrects_topk += prediction_analysis.count_corrects(label_trim, predicts_topk, label_lengths.long()-1, k=k)
                        running_size += torch.sum(label_lengths-1)
                else:
                    with torch.no_grad():
                        predicts_topk = cnn_rnn.sample(text, image, text_lengths, path_length=label_trim.shape[1], beam_width=k) # B, T, k
                        predicts = predicts_topk[:,:,0] # B, T
                        running_corrects += prediction_analysis.count_corrects(label_trim, predicts, label_lengths.long()-1)
                        running_corrects_topk += prediction_analysis.count_corrects(label_trim, predicts_topk, label_lengths.long()-1, k=k)
                        running_size += torch.sum(label_lengths-1)

            epoch_loss = running_loss / running_size
            epoch_acc = running_corrects / running_size
            epoch_acc_topk = running_corrects_topk / running_size
            time_elapsed = time.time() - since
            print('{} Loss:\t{:.4f} Top1 Acc: {:.4f} Top{} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(phase, epoch_loss, \
                epoch_acc, k, epoch_acc_topk, time_elapsed // 60, time_elapsed % 60))
            
            if SHOW_TOP_PREDICT_FREQ:
                prediction_analysis.show_top_label_frequency(k=10)
                prediction_analysis.show_top_predict_frequency(k=10)
                print()
            
        
        scheduler.step() 
        if SAVE_MODEL and (epoch + 1) % 5 == 0:
            torch.save(cnn_rnn.state_dict(), MODEL_PATH)

train()
