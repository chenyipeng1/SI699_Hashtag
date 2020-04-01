from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



#Normalize every sentence
def normalize_sentence(df, lang):
   sentence = df[lang].str.lower()
   sentence = sentence.str.replace('[^A-Za-z\s]+', '')
   sentence = sentence.str.normalize('NFD')
   sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
   return sentence


def read_file(loc):
   df = pd.read_csv(loc, delimiter='\t', header=None)
   return df



def process_data(loc,lang):
   df = read_file(loc)
   sentence=normalize_sentence(df, lang)

   source = Lang()
   for i in range(len(df)):
       if len(sentence[i].split(' ')) < MAX_LENGTH:
           source.addSentence(sentence[i])

   return source



def indexesFromSentence(lang, sentence):
   return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
   indexes = indexesFromSentence(lang, sentence)
   indexes.append(EOS_token)
   return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)



