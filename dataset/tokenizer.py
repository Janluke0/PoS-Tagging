import tokenizers
from .twtita import TWITADS
from pathlib import Path


def train_tokenizer_from_twita(split_name):
    def word_tokenizer(w): return [w]
    ds = TWITADS(split_name, word_tokenizer)
    sentences = [' '.join(w) for w, _ in ds]
    return train_tokenizer_from_iter(sentences)


def train_tokenizer_from_iter(sentences):
    my_tokenizer = tokenizers.ByteLevelBPETokenizer()
    # and train
    my_tokenizer.train_from_iterator(sentences, vocab_size=3000, min_frequency=3,
                                     special_tokens=['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[UNK]'])
    my_tokenizer.post_processor = tokenizers.processors.TemplateProcessing(single='<s> $A </s>',
                                                                           special_tokens=[('<s>', my_tokenizer.token_to_id('[BOS]')),
                                                                                           ('</s>', my_tokenizer.token_to_id('[EOS]'))])
    return my_tokenizer
