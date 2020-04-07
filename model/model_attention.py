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
    def __init__(self):
        """Load the pretrained ResNet-18 and replace top fully connected layer."""
        super(EncoderCNN, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]      # delete the last fc layer.
        self.resnet18 = nn.Sequential(*modules)
        
    def forward(self, images):
        """Extract feature vectors from input images.
            input: images ()
            output: features (N, H*W, C)
        """
        
        with torch.no_grad():
            features = self.resnet18(images)
        N,C,H,W=features.size()
        #print('features',features.size())
        features = features.view(N,C,H*W)
        features = features.permute(0,2,1)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        '''
        input: 
        vocab_size: dimension of label vectors
        embed_size: dimension of label embedding vectors
        hidden_size: dimension of lstm hidden states
        num_layers: number of layers in lstm
        '''
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)
        self.linear_text = nn.Linear(hidden_size, hidden_size)    #infeatures:hidden_size, outfeatures:vocab_size
        self.linear_image = nn.Linear(hidden_size, hidden_size)
        self.linear_label = nn.Linear(hidden_size, hidden_size)

        self.embed_size = embed_size
        self.hidden_size = hidden_size
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
        embeddings = self.embed(labels[:,:-1]) # B, T, label_embed_size
        image_features = torch.mean(image_features, 1) #.unsqueeze(1) # B, C
        #image_text_features = torch.cat((image_features, text_features), 1) # B, C + text_hidden_size
        #print(image_features.shape)
        batch_size, time_step = labels.size()
        time_step -= 1
        predicts = to_var(torch.zeros(batch_size, time_step, self.vocab_size))
        hx = to_var(torch.zeros(batch_size, self.hidden_size))
        cx = to_var(torch.zeros(batch_size, self.hidden_size))
        #features = torch.cat((image_text_features, torch.zeros(batch_size, text_features.shape[1]), 1) # B, C + text_hidden_size + text_hidden_size
        text_projection = self.linear_text(text_features) # B, hidden_size
        image_projection = self.linear_image(image_features) # B, hidden_size

        #### print projection layer
        # print(text_projection)
        # print('*' * 50)
        # print(image_projection)
        # print('*' * 50)
        #print("text projection ", text_projection.shape)
        #print("image projection ", image_projection.shape)
        for i in range(time_step): 
            #feas, _ = self.attention(features,hx)
            hx, cx = self.lstm_cell(embeddings[:,i,:], (hx, cx)) # B, hidden_size
            #print("hx ", hx.shape)
            xt = torch.sum(torch.stack((text_projection, image_projection, hx), dim=0), dim=0)  # B, hidden_size
            #xt = torch.sum(torch.stack((image_projection, hx), dim=0), dim=0)  # B, hidden_size
            #print("xt ", xt.shape)
            label_embedding = torch.mm(xt, self.embed.weight.transpose(1,0)) # B, vocab_size
            #print("label_embedding ", label_embedding.shape)
            predicts[:,i,:] = F.softmax(label_embedding, dim=1) #
        return predicts
    
    def sample(self, features, states=None):
        """Generate labels for given image features using greedy search."""
        sampled_ids = []
        hx=to_var(torch.zeros(1,1024))
        cx=to_var(torch.zeros(1,1024))
        inputs = torch.mean(features,1)
        alphas=[]
        for i in range(self.max_seg_length):
            feas,alpha=self.attention(features,hx)
            alphas.append(alpha)
            inputs=torch.cat((feas,inputs),-1)
            hx, cx = self.lstm_cell(inputs,(hx,cx))          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hx.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids,alphas
