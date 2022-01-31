from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForTokenClassification


class DistilBERTPos(nn.Module):
    """
        https://huggingface.co/Davlan/distilbert-base-multilingual-cased-ner-hrl
    """

    def __init__(self, nlabels):
        super(DistilBERTPos, self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained(
            "Davlan/distilbert-base-multilingual-cased-ner-hrl")
        self.pretrained.classifier = nn.Linear(
            self.pretrained.classifier.in_features, self.nlabels)

    def forward(self, *args, **kwargs):
        scores = self.pretrained(*args, **kwargs).logits
        return F.log_softmax(scores, dim=1)

    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("Davlan/distilbert-base-multilingual-cased-ner-hrl")


class RoBERTaXLMPos(nn.Module):
    """
        https://huggingface.co/Davlan/xlm-roberta-large-ner-hrl
        multilingual
    """

    def __init__(self, nlabels):
        super(RoBERTaXLMPos, self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained(
            "Davlan/xlm-roberta-large-ner-hrl")
        self.pretrained.classifier = nn.Linear(
            self.pretrained.classifier.in_features, self.nlabels)

    def forward(self, *args, **kwargs):
        scores = self.pretrained(*args, **kwargs).logits
        return F.log_softmax(scores, dim=1)

    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
