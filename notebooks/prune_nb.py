import sys, json, os
from pathlib import Path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        _dir = Path(sys.argv[1])
    else:
        _dir = Path(os.path.realpath(__file__)).parent
    for f in _dir.iterdir():
        print(f)
        try:
            if f.suffix == '.ipynb':
                with f.open() as nb:
                    parsed = json.loads(nb.read())
                parsed['metadata']['widgets'] = {}
                with f.open('w') as nb:
                    nb.write(json.dumps(parsed))
        except:
            print('err')