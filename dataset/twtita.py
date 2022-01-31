from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import re
import requests
import functools
_SCR = {
    'test':  "https://raw.githubusercontent.com/evalita2016/data/master/postwita/goldTESTset-2016_09_12.txt",
    'train': "https://raw.githubusercontent.com/evalita2016/data/master/postwita/goldDEVset-2016_09_05.txt"
}


class TWITADS(Dataset):
    _TAGS = {'ADJ': 0, 'ADP': 1, 'ADP_A': 2, 'ADV': 3,
            'AUX': 4, 'CONJ': 5, 'DET': 6, 'EMO': 7,
            'HASHTAG': 8, 'INTJ': 9, 'MENTION': 10,
            'NOUN': 11, 'NUM': 12, 'PART': 13, 'PRON': 14,
            'PROPN': 15, 'PUNCT': 16, 'SCONJ': 17, 'SYM': 18,
            'URL': 19, 'VERB': 20, 'VERB_CLIT': 21, 'X': 22,
            '[PAD]':-100}

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
                [t, *[self._TAGS['[PAD]']]*(len(w_tkns)-1)]
                ) for w_tkns, t in tweet]
        elif self.tag_mode=='last':
            tags = [np.array([*[self._TAGS['[PAD]']]*(len(w_tkns)-1), t]) for w_tkns, t in tweet]
        elif self.tag_mode=='terminal':
            tags = [np.array(
                    [t, *[self._TAGS['[PAD]']]*(len(w_tkns)-2), t]
                    if len(w_tkns)>1 else [t]
                ) for w_tkns, t in tweet]

        tags = np.stack(flatten(tags))
        tokens = np.array(flatten([w for w,_ in tweet]))

        if self.transform:
            return self.transform(tokens,tags)

        return tokens,tags


def flatten(t):
    return [item for sublist in t for item in sublist]

def get_ds(name):
    if name!="test" and name!="train":
        raise Exception("Invalid dataset")

    f = Path("twitads")
    f.mkdir(parents=True,exist_ok=True)
    f /= f"{name}.dat"

    if not f.exists():
        res = requests.get(_SCR[name])
        if not res.ok:
            raise Exception("Error downloading dataset")
        f.write_bytes(res.content)

    return parse_TWITA(f)

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
