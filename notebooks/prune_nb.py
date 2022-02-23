import sys, json, os
from pathlib import Path


def prune_in_dir(_dir, recursive=True):
    for f in _dir.iterdir():
        try:
            if f.suffix == '.ipynb':
                with f.open() as nb:
                    parsed = json.loads(nb.read())
                parsed['metadata']['widgets'] = {}
                with f.open('w') as nb:
                    nb.write(json.dumps(parsed))
            elif f.is_dir() and recursive:
                prune_in_dir(f)
        except:
            print('ERROR')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        _dir = Path(sys.argv[1])
    else:
        _dir = Path(os.path.realpath(__file__)).parent
    prune_in_dir(_dir)
