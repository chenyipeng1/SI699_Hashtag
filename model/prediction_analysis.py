import torch
import collections

class PredictionAnalysis():

    def __init__(self, label2tag):
        self.predict_counter = collections.defaultdict(int)
        self.label_counter = collections.defaultdict(int)
        self.label2tag = label2tag

    def count_corrects(self, labels, predicts, label_lengths, k=None):
        corrects = 0
        batch_size, time_step = labels.shape
        if not k:
            for i in range(batch_size):
                label = labels[i,:label_lengths[i]]
                predict = predicts[i,:label_lengths[i]]
                corrects += torch.sum(label == predict)
                for p in predict:
                    self.predict_counter[p.item()] += 1
                for l in label:
                    self.label_counter[l.item()] += 1
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
    
    def show_top_predict_frequency(self, k=10):
        print("Top ", k, " predictions with frequency")
        print([(self.label2tag[x[0]], x[1]) for x in sorted(self.predict_counter.items(), key=lambda x: x[1], reverse=True)[:k]])
    
    def show_top_label_frequency(self, k=10):
        print("Top ", k, " labels with frequency")
        print([(self.label2tag[x[0]], x[1]) for x in sorted(self.label_counter.items(), key=lambda x: x[1], reverse=True)[:k]])        