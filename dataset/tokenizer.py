import tokenizers
from .twtita import get_ds
from pathlib import Path
from transformers import AutoTokenizer


class UnsupportedTokenizer(Exception):
    pass


def get_tokenizer(split_name, _type="BPE", vocab_size=2048, directory="twitads/tokenizers"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if _type == 'BPE':
        file = directory / f"{split_name}_{_type}_{vocab_size}"
        if True:  # not file.exists():
            tknzr = train_tokenizer_from_twita(
                split_name, _type, vocab_size=vocab_size)
            file.mkdir(parents=True, exist_ok=True)
            tknzr.save_model(str(file))
            return tknzr  # skip saving 4now
        return tokenizers.ByteLevelBPETokenizer().from_file(*file.iterdir())
    else:
        raise UnsupportedTokenizer()


def train_tokenizer_from_twita(split_name, _type="BPE", **kwargs):
    _, tweets = get_ds(split_name)
    sentences = []
    for t in tweets:
        sentences.append(
            " ".join([word for word, _ in t]))
    if _type == "BPE":
        return train_bpe_from_iter(sentences, **kwargs)
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
