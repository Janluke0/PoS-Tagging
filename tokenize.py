from dataset.tokenizer import get_tokenizer
from dataset.twtita import mk_dataloaders
from pprint import pprint
if __name__ == '__main__':
    tnames = [
        'BPE', 'WordPiece', 'BERT_pretrained', 'DBERT_pretrained',
        'ELECTRA_pretrained', 'ROBERTA_pretrained'
    ]
    tokenizers = [(get_tokenizer('resampled_train', tname,
                                 vocab_size=2048), tname)
                  for tname in tnames[:1]]
    all_metrics = []
    for tk, name in tokenizers:
        metrics = {'name': name}
        metrics['vocab_size'] = (tk.vocab_size if hasattr(tk, 'vocab_size')
                                 else tk.get_vocab_size())
        _, dl = mk_dataloaders(tk,
                               ds_names=['resampled_train'],
                               batch_size=1,
                               shuffle=False,
                               align_labels=False)
        all_tokens = []
        tot = 0
        M, m = 0, float('inf')
        for x, _ in iter(dl):
            size = x.size(1)
            M = max(M, size)
            m = min(m, size)
            tot += size
            all_tokens.extend([i.int().item() for i in x[0]])
        metrics['avg_sentence_len'] = tot / len(dl)
        metrics['max_sentence_len'] = M
        metrics['min_sentence_len'] = m
        metrics['used_tokens'] = len(set(all_tokens))
        all_metrics.append(metrics)
    pprint(all_metrics)
    pass
