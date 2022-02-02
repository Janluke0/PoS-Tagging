from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForTokenClassification


class ItBERTCasedPos(nn.Module):
    """
        https://huggingface.co/dbmdz/bert-base-italian-cased
    """

    def __init__(self, nlabels):
        super(ItBERTCasedPos, self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-base-italian-cased")
        self.pretrained.classifier = nn.Linear(
            self.pretrained.classifier.in_features, self.nlabels)

    def forward(self, *args, **kwargs):
        scores = self.pretrained(*args, **kwargs)
        scores = scores.logits
        return F.log_softmax(scores, dim=1)

    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")


class ItBERTUncasedPos(nn.Module):
    """
        https://huggingface.co/dbmdz/bert-base-italian-uncased
    """

    def __init__(self, nlabels):
        super(ItBERTUncasedPos, self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-base-italian-uncased")
        self.pretrained.classifier = nn.Linear(
            self.pretrained.classifier.in_features, self.nlabels)

    def forward(self, *args, **kwargs):
        scores = self.pretrained(*args, **kwargs)
        scores = scores.logits
        return F.log_softmax(scores, dim=1)

    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")


class ItELECTRACasedPos(nn.Module):
    """
        https://huggingface.co/dbmdz/electra-base-italian-mc4-cased-discriminator
    """

    def __init__(self, nlabels):
        super(ItELECTRACasedPos, self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/electra-base-italian-mc4-cased-discriminator")
        self.pretrained.classifier = nn.Linear(
            self.pretrained.classifier.in_features, self.nlabels)

    def forward(self, *args, **kwargs):
        scores = self.pretrained(*args, **kwargs)
        scores = scores.logits
        return F.log_softmax(scores, dim=1)

    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("dbmdz/electra-base-italian-mc4-cased-discriminator")


class ItELECTRAXXLCasedPos(nn.Module):
    """
        https://huggingface.co/dbmdz/electra-base-italian-xxl-cased-discriminator
    """

    def __init__(self, nlabels):
        super(ItELECTRAXXLCasedPos, self).__init__()
        self.nlabels = nlabels
        self.pretrained = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/electra-base-italian-xxl-cased-discriminator")
        self.pretrained.classifier = nn.Linear(
            self.pretrained.classifier.in_features, self.nlabels)

    def forward(self, *args, **kwargs):
        scores = self.pretrained(*args, **kwargs)
        scores = scores.logits
        return F.log_softmax(scores, dim=1)

    @staticmethod
    def tokenizer():
        return AutoTokenizer.from_pretrained("dbmdz/electra-base-italian-xxl-cased-discriminator")
