from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from pathlib import Path
import numpy as np
import re
import requests
import functools
from random import Random

_SRC = {
    'test':  "https://raw.githubusercontent.com/evalita2016/data/master/postwita/goldTESTset-2016_09_12.txt",
    'train': "https://raw.githubusercontent.com/evalita2016/data/master/postwita/goldDEVset-2016_09_05.txt"
}


class TWITADS(Dataset):
    _TAGS = {'[PAD]': 0,  # PAD To be ignored
             'ADJ': 1, 'ADP': 2,
             'ADP_A': 3, 'ADV': 4, 'AUX': 5,
             'CONJ': 6, 'DET': 7, 'EMO': 8,
             'HASHTAG': 9, 'INTJ': 10, 'MENTION': 11,
             'NOUN': 12, 'NUM': 13, 'PART': 14,
             'PRON': 15, 'PROPN': 16, 'PUNCT': 17,
             'SCONJ': 18, 'SYM': 19, 'URL': 20,
             'VERB': 21, 'VERB_CLIT': 22, 'X': 23,
             '[BOS]': 24, '[EOS]': 25,
             '[EPAD]': 26,  # Explicit PAD
             '[MASK]': 27, '[UNS1]': 28, '[UNS2]': 29
             }

    def __init__(self, dataset, word_tokenizer, tag_mode="all", transform=None):
        """
            Parameters
            ----------
                dataset : Union["test", "train"]

                word_tokenizer : Callable[[Iterable[str]], Iterable[int]]
                    Given a sequence of word returns their tokenization
                    and the index of the firt token per each word

                tag_mode: Union["all","first","last","terminal"]
                    How a word tag is associated with the tokens of the word:
                        all: for each token
                        first: the first token (for the others is [PAD])
                        last: the last token
                        terminal: the first and last tokens

                transform: Callable[[np.ndarray[Iterable[Int],Iterable[Int]],np.ndarray],Any]
        """
        super(TWITADS,self).__init__()
        self.word_tokenizer = word_tokenizer
        self.ids, self.tweets = get_ds(dataset)
        self.n_tags = len(self._TAGS)
        self.tag_mode = tag_mode.lower()
        self.transform = transform

    def __len__(self):
        return len(self.ids)


    @functools.lru_cache(None)
    def __getitem__(self, idx):
        tweet = [(self.word_tokenizer(w), self._TAGS[t])
            for w,t in self.tweets[idx]]

        if self.tag_mode=='all':
            tags = [np.array([t]*len(w_tkns)) for w_tkns, t in tweet]
        elif self.tag_mode=='first':
            tags = [np.array(
                [t, *[self._TAGS['[EPAD]']]*(len(w_tkns)-1)]
                ) for w_tkns, t in tweet]
        elif self.tag_mode=='last':
            tags = [np.array([*[self._TAGS['[EPAD]']]*(len(w_tkns)-1), t]) for w_tkns, t in tweet]
        elif self.tag_mode=='terminal':
            tags = [np.array(
                    [t, *[self._TAGS['[EPAD]']]*(len(w_tkns)-2), t]
                    if len(w_tkns)>1 else [t]
                ) for w_tkns, t in tweet]

        tags = np.stack(flatten(tags))
        tokens = np.array(flatten([w for w,_ in tweet]))

        if self.transform:
            return self.transform(tokens,tags)

        return tokens,tags


def tokenize(tokenizer, tokens):
    add_special_token = True
    try:
        tokenized_inputs = tokenizer.encode(tokens, is_pretokenized=True)
        word_ids = tokenized_inputs.word_ids
        ids = tokenized_inputs.ids
    except:
        tmp = tokenizer(tokens, is_split_into_words=True,
                        add_special_tokens=False,
                        return_attention_mask=False, return_token_type_ids=False)
        add_special_token = False
        word_ids = tmp.word_ids()
        ids = tmp['input_ids']
    return word_ids, ids, add_special_token


def tokenize_and_align_labels(tokenizer,
                              tokens,
                              tags,
                              epad_subtokens=True,
                              align_labels=True):
    tokens = [" " + w if i > 0 else w for i, w in enumerate(list(tokens))]

    word_ids, ids, add_special_token = tokenize(tokenizer, tokens)
    # Map tokens to their respective word.
    previous_word_idx = None
    label_ids = []
    if add_special_token:
        label_ids.append(TWITADS._TAGS['[BOS]'])
    for word_idx in word_ids:
        if word_idx is None:
            pass
        elif word_idx != previous_word_idx or not epad_subtokens:
            label_ids.append(tags[word_idx])
        elif align_labels:
            label_ids.append(TWITADS._TAGS['[EPAD]'])
        previous_word_idx = word_idx
    if add_special_token:
        label_ids.append(TWITADS._TAGS['[EOS]'])

    return torch.tensor(ids), torch.tensor(label_ids)


def collate_fn(batch):
    tokens, tags = zip(*batch)
    return pad_sequence(tokens, batch_first=True), pad_sequence(tags, batch_first=True)


def mk_dataloaders(tknzr,
                   ds_names=['train', 'test'],
                   batch_size=64,
                   shuffle=True,
                   align_labels=True,
                   epad_subtokens=True):

    def transformer(tkns, tags):
        return tokenize_and_align_labels(tknzr, tkns, tags, epad_subtokens,
                                         align_labels)

    def word_tokenizer(w): return [w]

    dataloaders = []

    for name in ds_names:
        ds = TWITADS(name, word_tokenizer,
                     transform=transformer)
        dataloaders.append(DataLoader(ds, shuffle=shuffle,
                                      batch_size=batch_size, collate_fn=collate_fn))
    return (ds.n_tags, *dataloaders)

def flatten(t):
    return [item for sublist in t for item in sublist]


def get_ds(name, _dir=Path("twitads")):
    _dir.mkdir(parents=True, exist_ok=True)
    f = _dir/f"{name}.dat"

    if not f.exists() and name not in _SRC:
        raise Exception("Invalid dataset")

    if not f.exists():
        res = requests.get(_SRC[name])
        if not res.ok:
            raise Exception("Error downloading dataset")
        f.write_bytes(res.content)

    return parse_TWITA(f)


def write_ds(fname, ids, tweets, _dir=Path("twitads")):
    assert len(ids) == len(tweets)
    with (_dir/fname).open("w+", encoding='utf8') as f:
        for i in range(len(ids)):
            f.write(f"_____{ids[i]}_____\n")
            for word, tag in tweets[i]:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")


def split_ds(ids, tweets, percentage=0.9, seed=42):
    r = Random(seed)
    resampled_A_ids = r.sample(ids, int(len(ids)*percentage))
    resampled_B_ids = list(filter(lambda i: i not in resampled_A_ids, ids))
    resampled_A_tweets = [tweets[i]
                          for i in [ids.index(_id) for _id in resampled_A_ids]]
    resampled_B_tweets = [tweets[i]
                          for i in [ids.index(_id) for _id in resampled_B_ids]]
    return (resampled_A_ids, resampled_A_tweets), (resampled_B_ids, resampled_B_tweets)

def parse_TWITA(fname):
    fixes = {'ADV2':'ADP_A', 'STYM':'SYM', 'AD':'ADJ'}#fix errors in test
    data = open(fname, encoding ="utf8").read()
    tweets = [row.strip() for row in re.split('_____\d+_____\n',data) if not row.strip() == "" ]
    tweets = [tuple((*word.split(),) for word in tweet.split('\n')) for tweet in tweets]
    ids = [v.strip().replace('_','') for v in re.findall('_____\d+_____\n',data)]
    out_tweets = []
    for tweet in tweets:
        out_tweets.append([(tw,(tg if tg not in fixes else fixes[tg])) for tw,tg in tweet])
    return ids, out_tweets
