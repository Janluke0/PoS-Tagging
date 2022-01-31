from genericpath import exists
from pathlib import Path
import itertools
from xml.etree.ElementInclude import include

from bpemb import BPEmb

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import tqdm

from dataset.twtita import TWITADS
from model.lstm import LSTMTagger
from model import train_model
from pprint import pprint

TRAIN = True
SEED = 42
DROPOUT, N_TOKENS, BATCH_SIZE, CUDA = .1, 1000, 128, torch.cuda.is_available()

def collate_fn(batch):
    tokens, tags = zip(*batch)
    return pad_sequence(tokens,batch_first=True), pad_sequence(tags, padding_value=-100,batch_first=True)

bpe = BPEmb(lang='it',vs=N_TOKENS)
def mk_dl(special,tag_mode):
    if special == '#ow':
        word_tokenizer = lambda word: [1,*bpe.encode_ids(word),2]
    elif special == 'eow':
        word_tokenizer = lambda word: [*bpe.encode_ids(word),2]
    if special == 'bow':
        word_tokenizer = lambda word: [1,*bpe.encode_ids(word)]
    elif special == '':
        word_tokenizer = lambda word: bpe.encode_ids(word)

    transformer = lambda tkns,tags:  (torch.tensor(tkns), torch.tensor(tags).long())

    ds_train = TWITADS('train',word_tokenizer, transform=transformer, tag_mode=tag_mode)
    ds_test  = TWITADS('test',word_tokenizer, transform=transformer, tag_mode=tag_mode)
    return (
        ds_train.n_tags,
        DataLoader(ds_train, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn),
        DataLoader(ds_test, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    )


bi = [True, False]
hidden_dim = [16,32,64,128]
output_layers = [1]#,2,3] we can reuse lstm with multiple layers
lstm_layers = [1,2]
special_tokens = ['bow','eow','#ow','']
tag_mode = ['last','all','first','terminal']

all_params_comb = list(itertools.product(lstm_layers,hidden_dim,output_layers,special_tokens,tag_mode, bi))

models = []

models_dir = Path("trained")
models_dir.mkdir(exist_ok=True)


pbar = tqdm.tqdm(all_params_comb[::-1])
for l_layers, hid_dim, o_layers, special_tkns, tg_mode, is_bi in pbar:
    k = f"{'bi' if is_bi else 'mono'}_{l_layers}_{hid_dim}_{o_layers}_{special_tkns}_{tg_mode}"
    pbar.set_description(k)
    if (models_dir/f"{k}.pth").exists():
        continue

    n_tags, dl_train,dl_test = mk_dl(special_tkns, tg_mode)

    m = LSTMTagger(
        N_TOKENS,
        n_tags,
        hidden_dim=hid_dim,
        dropout=DROPOUT,
        lstm_layers=l_layers,
        bidirectional=is_bi,
        output_layers=o_layers
    )

    if CUDA:
            m.to('cuda')

    models.append(m)

    torch.manual_seed(SEED)
    loss, acc = train_model(m, dl_train,dl_test, CUDA, epochs=100,lr=.1)

    torch.save(m.state_dict(), models_dir/f"{k}.pth")
    with open(models_dir/f"{k}.csv","w") as f:
        f.write(','.join((str(l) for l in loss)))
        f.write('\n')
        f.write(','.join((str(a) for a in acc)))
