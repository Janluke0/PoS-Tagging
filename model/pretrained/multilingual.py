
from model.pretrained import BasePretrained


class DistilBERT(BasePretrained):
    """
        https://huggingface.co/Davlan/distilbert-base-multilingual-cased-ner-hrl
    """
    def __init__(self, d_output):
        super(type(self), self).__init__(d_output,"Davlan/distilbert-base-multilingual-cased-ner-hrl")

class RoBERTaXLM(BasePretrained):
    """
        https://huggingface.co/Davlan/xlm-roberta-large-ner-hrl
    """
    def __init__(self, d_output):
        super(type(self), self).__init__(d_output,"Davlan/xlm-roberta-large-ner-hrl")