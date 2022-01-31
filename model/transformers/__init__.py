import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
import numpy as np
import re

from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW


class DistilBERTPos(nn.Module):
    """
        https://huggingface.co/Davlan/distilbert-base-multilingual-cased-ner-hrl
    """

    def __init__(self, nlabels):
        super(DistilBERTPos,self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained("Davlan/distilbert-base-multilingual-cased-ner-hrl")
        self.pretrained.classifier = nn.Linear(self.pretrained.classifier.in_features,self.nlabels)

    def forward(self, *args,**kwargs):
        scores = self.pretrained(*args,**kwargs).logits
        return F.log_softmax(scores,dim=1)

    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("Davlan/distilbert-base-multilingual-cased-ner-hrl")


class RoBERTaXLMPos(nn.Module):
    """
        https://huggingface.co/Davlan/xlm-roberta-large-ner-hrl
        multilingual
    """
    def __init__(self, nlabels):
        super(RoBERTaXLMPos,self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
        self.pretrained.classifier = nn.Linear(self.pretrained.classifier.in_features,self.nlabels)

    def forward(self, *args,**kwargs):
        scores = self.pretrained(*args,**kwargs).logits
        return F.log_softmax(scores,dim=1)

    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")


class ItBERTCasedPos(nn.Module):
    """
        https://huggingface.co/dbmdz/bert-base-italian-cased
    """
    def __init__(self, nlabels):
        super(ItBERTCasedPos,self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-italian-cased")
        self.pretrained.classifier = nn.Linear(self.pretrained.classifier.in_features,self.nlabels)

    def forward(self, *args,**kwargs):
        scores = self.pretrained(*args,**kwargs)
        scores = scores.logits
        return F.log_softmax(scores,dim=1)


    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")



class ItBERTUncasedPos(nn.Module):
    """
        https://huggingface.co/dbmdz/bert-base-italian-uncased
    """
    def __init__(self, nlabels):
        super(ItBERTCasedPos,self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-italian-uncased")
        self.pretrained.classifier = nn.Linear(self.pretrained.classifier.in_features,self.nlabels)

    def forward(self, *args,**kwargs):
        scores = self.pretrained(*args,**kwargs)
        scores = scores.logits
        return F.log_softmax(scores,dim=1)


    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")