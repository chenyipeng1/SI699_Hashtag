import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models, transforms
import time
import os
import copy
from dataloader import TweetData
from model_attention import EncoderCNN
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

tweet_data = TweetData(batch_size=8, file_size=None)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using ", device)
model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, tweet_data.label_generator.label_num)
model_ft = model_ft.to(device)
epoch_num = 20    
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()
MODEL_PATH = "/home/feiyi/SI699_Hashtag/serialized/img_baseline.pt"

for epoch in range(epoch_num):
    
    print('Epoch {}/{}'.format(epoch+1, epoch_num))
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0
    running_size = 0.0
    phase = None

    for phase in ["train", "val", "test"]:
        since = time.time()
        for batch_data in tweet_data.dataloaders[phase]:
            #text = batch_data["text"].to(device)
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model_ft(image) # B, C
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
torch.save(model_ft.state_dict(), MODEL_PATH)
# criterion = nn.CrossEntropyLoss()

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=20)
