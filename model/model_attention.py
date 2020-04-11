import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import avg_pool2d
from torch.autograd import Variable

#Variable has been deprecated for now
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class EncoderCNN(nn.Module):
    def __init__(self, resnet_version="resnet18", train_resnet=False):
        """Load the pretrained ResNet and replace top fully connected layer."""
        super(EncoderCNN, self).__init__()
        resnet = None
        if resnet_version == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif resnet_version == "resnet34":
            resnet = models.resnet34(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        self.image_hidden_size = list(resnet.children())[-1].in_features
        print("Using ", resnet_version, " with output feature size ", self.image_hidden_size)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.train_resnet = train_resnet
        
        
    def forward(self, images):
        """Extract feature vectors from input images.
            input: images ()
            output: features (N, H*W, C)
        """
        features = None
        if self.train_resnet:
            features = self.resnet(images)
        else:
            with torch.no_grad():
                features = self.resnet(images)
        N, C, H, W = features.size()
        #print('features',features.size())
        features = features.view(N, C, H * W)
        features = features.permute(0, 2, 1)
        return features



class DecoderRNN(nn.Module):
    def __init__(self, image_hidden_size, text_hidden_size, label_hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        '''
        input: 
        vocab_size: dimension of label vectors
        hidden_size: dimension of lstm hidden states
        num_layers: number of layers in lstm
        '''
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, label_hidden_size, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(label_hidden_size, label_hidden_size)
        self.linear_text = nn.Linear(text_hidden_size, label_hidden_size)    #infeatures:hidden_size, outfeatures:vocab_size
        self.linear_image = nn.Linear(image_hidden_size, label_hidden_size)

        self.label_hidden_size = label_hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.vis_dim=512
        self.hidden_dim=1024
        vis_num=196
        self.att_vw = nn.Linear(self.vis_dim,self.vis_dim,bias=False)
        self.att_hw = nn.Linear(self.hidden_dim,self.vis_dim,bias=False)
        self.att_bias = nn.Parameter(torch.zeros(vis_num))
        self.att_w = nn.Linear(self.vis_dim,1,bias=False)

    def attention(self,features,hiddens):
        '''
        attention of RNN model, changes when predicting different labels
        each step: the average of the synthesized image of all the label nodes at the softmax layer
        input: image features ,hiddens: (batch_size, 1, hidden_size)
        output: sum of all elements based on feature vector row, rescaled att_out
        '''
        att_fea = self.att_vw(features)
        att_h = self.att_hw(hiddens).unsqueeze(1)
        att_full = nn.ReLU()(att_fea + att_h +self.att_bias.view(1,-1,1))
        att_out = self.att_w(att_full)
        alpha=nn.Softmax(dim=1)(att_out)
        context=torch.sum(features*alpha,1) 
        return context,alpha

    def forward(self, text_features, image_features, labels):
        """Decode image feature vectors and generates labels."""
        '''
        input: extracted text_features (B, text_hidden_size), image features (B, H*W, C), labels (B, T)
        output: predicted label 
        '''
        embeddings = self.embed(labels[:,:-1]) # B, T, label_hidden_size
        image_features = torch.mean(image_features, 1) # B, image_hidden_size
        #image_text_features = torch.cat((image_features, text_features), 1) # B, C + text_hidden_size
        #print(image_features.shape)
        batch_size, time_step = labels.size()
        time_step -= 1
        predicts = to_var(torch.zeros(batch_size, time_step, self.vocab_size))
        hx = to_var(torch.zeros(batch_size, self.label_hidden_size))
        cx = to_var(torch.zeros(batch_size, self.label_hidden_size))
        #features = torch.cat((image_text_features, torch.zeros(batch_size, text_features.shape[1]), 1) # B, C + text_hidden_size + text_hidden_size
        text_projection = self.linear_text(text_features) # B, label_hidden_size
        image_projection = self.linear_image(image_features) # B, label_hidden_size

        for i in range(time_step): 
            #feas, _ = self.attention(features,hx)
            hx, cx = self.lstm_cell(embeddings[:,i,:], (hx, cx)) # B, label_hidden_size
            #print("hx ", hx.shape)
            xt = torch.sum(torch.stack((text_projection, image_projection, hx), dim=0), dim=0)  # B, label_hidden_size
            #xt = torch.sum(torch.stack((image_projection, hx), dim=0), dim=0)  # B, label_hidden_size
            #print("xt ", xt.shape)
            label_embedding = torch.mm(xt, self.embed.weight.transpose(1,0)) # B, vocab_size
            #print("label_embedding ", label_embedding.shape)
            predicts[:,i,:] = label_embedding
        return predicts
    
    def sample_greedy(self, text_features, image_features, path_length=10):
        """Generate labels for given image features using greedy search."""
        batch_size = text_features.shape[0]
        time_step = path_length

        predicts = to_var(torch.zeros(batch_size, time_step))
        hx = to_var(torch.zeros(batch_size, self.label_hidden_size))
        cx = to_var(torch.zeros(batch_size, self.label_hidden_size))

        image_features = torch.mean(image_features, 1) # B, image_hidden_size
        text_projection = self.linear_text(text_features) # B, label_hidden_size
        image_projection = self.linear_image(image_features) # B, label_hidden_size

        label = torch.ones((batch_size)).long() # B     1 as <start>
        

        for i in range(time_step):
            embeddings = self.embed(label) # B, label_hidden_size
            hx, cx = self.lstm_cell(embeddings, (hx, cx)) # B, label_hidden_size
            xt = torch.sum(torch.stack((text_projection, image_projection, hx), dim=0), dim=0)
            label_embedding = torch.mm(xt, self.embed.weight.transpose(1,0)) # B, vocab_size
            _, label= torch.max(label_embedding, dim=1) # B
            predicts[:,i] = label

        return predicts
