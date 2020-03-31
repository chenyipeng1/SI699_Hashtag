from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from text_encoder import EncoderRNN

#define a class TextVocabulary that stores each word from each tweet
class TextVocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<EOS>": 0}
        self.word2count = {}
        self.index2word = {0: "<EOS>"}
        self.n_words = 1 

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

    def tokenize(self, s):
        # normalize(preprocess) the text
        regex_str = [
            r'<[^>]+>', # HTML tags
            r'(?:@[\w_]+)', # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
        
            r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
            r'(?:[\w_]+)', # other words
            r'(?:\S)+' # anything else
        ]

        tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
        return tokens_re.findall(s)

    def preprocess(self, s, lowercase=True):
        tokens = self.tokenize(s)
        # print(tokens)
        tokens = [token.lower() for token in tokens]

        html_regex = re.compile('<[^>]+>')
        tokens = [token for token in tokens if not html_regex.match(token)]
        
        mention_regex = re.compile('(?:@[\w_]+)')
        tokens = ['@user' if mention_regex.match(token) else token for token in tokens]
        
        url_regex = re.compile('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
        tokens = ['!url' if url_regex.match(token) else token for token in tokens]

        hashtag_regex = re.compile("(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")
        tokens = ['' if hashtag_regex.match(token) else token for token in tokens]
        return ' '.join([t for t in tokens if t]).replace('rt @user : ','')


    # Normalize every sentence
    # def normalize_sentence(df, lang):
    #     sentence = df[lang].str.lower()
    #     sentence = sentence.str.replace('[^A-Za-z\s]+', '')
    #     sentence = sentence.str.normalize('NFD')
    #     sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    #     return sentence


    # def read_file(loc):
    #    df = pd.read_csv(loc, delimiter='\t', header=None)
    #    return df

    # def process_data(loc,lang):
    #    df = read_file(loc)
    #    sentence=normalize_sentence(df, lang)

    #    source = Lang()
    #    for i in range(len(df)):
    #        if len(sentence[i].split(' ')) < MAX_LENGTH:
    #            source.addSentence(sentence[i])

    #    return source


    # convert sentence to index
    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]

    # convert sentence to tensor
    # for each sentence from the batch,use this function to convert it to tensor, and go through the encoder
    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(0)
        return torch.tensor(indexes, dtype=torch.long)#.view(-1, 1)



if __name__ == "__main__":

    tweet_data = TweetData(batch_size=2)
    source = TextVocabulary('tweet')

    for batch_data in tweet_data.dataloaders["train"]:
        # for each sentence from the batch data, first use the add_sentence function from the class Lang

        print("text: ", batch_data["text"])
        for sentence in batch_data["text"]:
            s=preprocess(sentence)
            source.addSentence(s)
        
        break

    print(source.n_words)
    t=tensorFromSentence(source,s)
    encoder=EncoderRNN(source.n_words, 2).to(device)
    encoder_hidden = encoder.initHidden()

    output=encoder(t[0],encoder_hidden)
    print(output)

    # sample train process:
    # encoder_hidden = encoder.initHidden()
    # encoder_optimizer.zero_grad()
    # input_length = input_tensor.size(0)
    # target_length = target_tensor.size(0)
    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # loss = 0

    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(
    #         input_tensor[ei], encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0, 0]

    #..........




