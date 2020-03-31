import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# class Encoder(nn.Module):
#    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
#        super(Encoder, self).__init__()
      
#        #set the encoder input dimesion , embbed dimesion, hidden dimesion, and number of layers 
#        self.input_dim = input_dim
#        self.embbed_dim = embbed_dim
#        self.hidden_dim = hidden_dim
#        self.num_layers = num_layers

#        #initialize the embedding layer with input and embbed dimention
#        self.embedding = nn.Embedding(input_dim, self.embbed_dim)
#        #intialize the GRU to take the input dimetion of embbed, and output dimention of hidden and
#        #set the number of gru layers
#        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
              
#    def forward(self, src):
      
#        embedded = self.embedding(src).view(1,1,-1)
#        outputs, hidden = self.gru(embedded)
#        return outputs, hidden


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
    
        # imput_size is the number of words in the class Lang defined in ../src/read_text.py. For example: input_lang.n_words

        # sample usage:        
        # hidden_size = 256
        # encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)


        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, texts, text_lengths):
        # input: texts (B, T), text_lengths (B)
        # output: text_features (B, hidden_size) 
        batch_size, _ = texts.shape
        embedded = self.embedding(texts)
        # embedded (B, T, text_vocab_size)
        embedded_packed = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(embedded_packed, self.initHidden(batch_size))
        output_padded = pad_packed_sequence(output, batch_first=True, padding_value=0)[0] # (B, T, hidden_size) 
        mask = (text_lengths-1).unsqueeze(1).unsqueeze(2).expand(-1,-1,self.hidden_size).long()
        # return features of last timestep
        return output_padded.gather(1, mask).squeeze()

    def initHidden(self, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
