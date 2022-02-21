from model.pretrained import BasePretrained


class BERTCased(BasePretrained):
    """
        https://huggingface.co/dbmdz/bert-base-italian-cased
    """

    def __init__(self, d_output):
        super(type(self), self).__init__(
            d_output, "dbmdz/bert-base-italian-cased")


class BERTUncased(BasePretrained):
    """
        https://huggingface.co/dbmdz/bert-base-italian-uncased
    """

    def __init__(self, d_output):
        super(type(self), self).__init__(
            d_output, "dbmdz/bert-base-italian-uncased")
