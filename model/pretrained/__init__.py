from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForTokenClassification


class BasePretrained(nn.Module):
    def __init__(self, d_output, model_name):
        super(type(self), self).__init__()
        self.d_output = d_output
        self.model_name = model_name
        self.pretrained = AutoModelForTokenClassification.from_pretrained(
            model_name)
        self.pretrained.classifier = nn.Linear(
            self.pretrained.classifier.in_features, self.nlabels)

    def forward(self, *args, **kwargs):
        scores = self.pretrained(*args, **kwargs).logits
        return F.log_softmax(scores, dim=1)

    @property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
