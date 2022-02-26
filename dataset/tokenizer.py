import tokenizers
from .twtita import get_ds
from pathlib import Path
from transformers import AutoTokenizer


class UnsupportedTokenizer(Exception):
    pass


def get_tokenizer(split_name,
                  _type="BPE",
                  vocab_size=2048,
                  directory="twitads/tokenizers",
                  **kwargs):

    # skip saving and load 4now
    if False:  # not file.exists():
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f"{split_name}_{_type}_{vocab_size}"
        file.mkdir(parents=True, exist_ok=True)
        #tknzr.save_model(str(file))

    tknzr = train_tokenizer_from_twita(split_name,
                                       _type,
                                       vocab_size=vocab_size,
                                       **kwargs)
    return tknzr


def train_tokenizer_from_twita(split_name, _type="BPE",vocab_size=2048, **kwargs):
    _, tweets = get_ds(split_name)
    sentences = []
    for t in tweets:
        sentences.append(
            " ".join([word for word, _ in t]))
    if _type == "BPE":
        return train_bpe_from_iter(sentences, vocab_size, **kwargs)
    elif _type == "WordPiece":
        return train_wordpiece_from_iter(sentences, vocab_size, **kwargs)
    elif _type == "BERT_pretrained":
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
    elif _type == "ELECTRA_pretrained":
        return AutoTokenizer.from_pretrained("dbmdz/electra-base-italian-mc4-cased-discriminator")
    elif _type == "ROBERTA_pretrained":
        return AutoTokenizer.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
    elif _type == "DBERT_pretrained":
        return AutoTokenizer.from_pretrained("Davlan/distilbert-base-multilingual-cased-ner-hrl")
    elif _type == 'BPEmb_pretrained':
        return BPEmbWrapper(**kwargs)
    else:
        raise UnsupportedTokenizer()


def train_bpe_from_iter(sentences,
                        vocab_size=2048, min_frequency=3,
                        special_tokens=['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[UNK]'], **kwargs):
    my_tokenizer = tokenizers.ByteLevelBPETokenizer()
    # and train
    my_tokenizer.train_from_iterator(sentences, vocab_size=vocab_size, min_frequency=min_frequency,
                                     special_tokens=special_tokens, **kwargs)

    my_tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single='[BOS] $A [EOS]',
        special_tokens=[
            ('[BOS]', my_tokenizer.token_to_id('[BOS]')),
            ('[EOS]', my_tokenizer.token_to_id('[EOS]'))
        ]
    )
    return my_tokenizer


def train_wordpiece_from_iter(sentences,
                              vocab_size=2048, min_frequency=3,
                              special_tokens=['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[UNK]'], **kwargs):
    my_tokenizer = tokenizers.BertWordPieceTokenizer()
    # and train
    my_tokenizer.train_from_iterator(sentences, vocab_size=vocab_size, min_frequency=min_frequency,
                                     special_tokens=special_tokens, **kwargs)

    my_tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single='[BOS] $A [EOS]',
        special_tokens=[
            ('[BOS]', my_tokenizer.token_to_id('[BOS]')),
            ('[EOS]', my_tokenizer.token_to_id('[EOS]'))
        ]
    )
    return my_tokenizer


from bpemb import BPEmb
from collections import namedtuple

TokenizedSentence = namedtuple("TokenizedSentence", ['word_ids', 'ids'])


class BPEmbWrapper:

    def __init__(self, *args, **kwargs):
        self.bpemb = BPEmb(*args, **kwargs)

    def encode(self, words, is_pretokenized=True):
        word_ids, ids = [self.bpemb.BOS], [None]
        for i, w in enumerate(words):
            tmp = self.bpemb.encode_ids(w)
            ids.extend([i] * len(tmp))
            word_ids.extend(tmp)

        word_ids.append(self.bpemb.EOS)
        ids.append(None)

        return TokenizedSentence(word_ids, ids)

    def __getattr__(self, name):
        return getattr(self.bpemb, name)

    @property
    def vocab_size(self):
        return self.bpemb.vs
