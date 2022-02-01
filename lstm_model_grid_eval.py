from scipy.optimize import curve_fit
from genericpath import exists
from pathlib import Path
import itertools

from bpemb import BPEmb

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import tqdm.auto as tqdm

from torch import nn

import numpy as np

from dataset.twtita import TWITADS
from model.lstm import LSTMTagger
from model import eval_model
from pprint import pprint
from sklearn.metrics import f1_score, explained_variance_score
from scipy.optimize import curve_fit

TRAIN = True
SEED = 42
DROPOUT, N_TOKENS, BATCH_SIZE, CUDA = .1, 1000, 128, torch.cuda.is_available()


def collate_fn(batch):
    tokens, tags = zip(*batch)
    return pad_sequence(tokens, batch_first=True), pad_sequence(tags, padding_value=-100, batch_first=True)


bpe = BPEmb(lang='it', vs=N_TOKENS)


def mk_dl(special, tag_mode, ds_names=['train', 'test']):
    if special == '#ow':
        def word_tokenizer(word): return [1, *bpe.encode_ids(word), 2]
    elif special == 'eow':
        def word_tokenizer(word): return [*bpe.encode_ids(word), 2]
    if special == 'bow':
        def word_tokenizer(word): return [1, *bpe.encode_ids(word)]
    elif special == '':
        def word_tokenizer(word): return bpe.encode_ids(word)

    def transformer(tkns, tags): return (
        torch.tensor(tkns), torch.tensor(tags).long())

    ds_train = TWITADS(ds_names[0], word_tokenizer,
                       transform=transformer, tag_mode=tag_mode)
    ds_test = TWITADS(ds_names[1], word_tokenizer,
                      transform=transformer, tag_mode=tag_mode)
    return (
        ds_train.n_tags,
        DataLoader(ds_train, shuffle=True,
                   batch_size=BATCH_SIZE, collate_fn=collate_fn),
        DataLoader(ds_test, shuffle=True,
                   batch_size=BATCH_SIZE, collate_fn=collate_fn)
    )


def mk_from_key(key, ds_names=['train', 'test']):
    is_bi, l_layers, hid_dim, o_layers, special_tkns, tg_mode = key.split('_')

    is_bi, l_layers, hid_dim, o_layers = is_bi == 'bi', int(
        l_layers), int(hid_dim), int(o_layers)
    n_tags, dl_tr, dl_te = mk_dl(special_tkns, tg_mode, ds_names)
    m = LSTMTagger(
        N_TOKENS,
        n_tags,
        hidden_dim=hid_dim,
        dropout=DROPOUT,
        lstm_layers=l_layers,
        bidirectional=is_bi,
        output_layers=o_layers
    )

    return m, dl_tr, dl_te


def compute_alpha(fname):
    """
        A made-up early performace metric:
            Assuming that loss go like an exp(-x/alpha)
            a lower value of alpha mean that is closer to convergence.
            Alpha is estimated applying non-linear least square
    """
    def curve(x, alpha):
        return np.exp(-x/alpha)
    with open(fname, 'r') as f:
        loss = list(map(float, f.readline().split(',')))
    xdata, ydata = np.array(range(100)), np.array(loss)
    m, M = ydata.min(), ydata.max()
    ydata = (ydata-m)/(M-m)
    popt, _ = curve_fit(curve, xdata, ydata)
    return popt[0]


models = []

models_dir = Path("trained")
models_dir.mkdir(exist_ok=True)
report_file = models_dir/"lstm_grid.csv"
loss_function = nn.NLLLoss()

with report_file.open("w+") as f:
    f.write("key,is_bi,lstm_layers,lstm_hidden_dim,out_layers,special_tokens,tag_mode,loss,accuracy,f1,explained_variance,alpha\n")
    for file in tqdm.tqdm(models_dir.iterdir()):
        if file.parts[-1].endswith('.pth'):
            k = file.parts[-1].replace('.pth', '')
            is_bi, l_layers, hid_dim, o_layers, special_tkns, tg_mode = k.split(
                '_')

            is_bi, l_layers, hid_dim, o_layers = is_bi == 'bi', int(
                l_layers), int(hid_dim), int(o_layers)
            model, _, dl_test = mk_from_key(k)
            model.load_state_dict(torch.load(file))
            model.eval()
            model.cuda()
            los, acc, y_true, y_pred = eval_model(
                model, dl_test, True, loss_function, return_y=True)
            f1 = f1_score(y_true, y_pred, average='macro')
            evs = explained_variance_score(
                y_true, y_pred, multioutput='variance_weighted')
            alpha = compute_alpha(models_dir/f"{k}.csv")
            f.write(",".join(map(str, (k, is_bi, l_layers, hid_dim,
                    o_layers, special_tkns, tg_mode, los, acc, f1, evs, alpha))))
            f.write("\n")
