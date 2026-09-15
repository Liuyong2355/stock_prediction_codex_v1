"""Restore accepted LF/CRLF bytes after checkout; never change frozen content/hashes."""
import argparse
import hashlib
import json
from pathlib import Path


TRAINING_SOURCES = ['e003.py','targets.py','e004_e005.py','baselines.py','folds.py',
                    'compare_official.py','diagnose_e002.py']


def accepted_text_bytes(current, expected_sha256):
    lf=current.replace(b'\r\n',b'\n')
    for candidate in [current,lf,lf.replace(b'\n',b'\r\n')]:
        if hashlib.sha256(candidate).hexdigest()==expected_sha256:
            return candidate
    raise ValueError('Content differs from accepted hash; LF/CRLF restoration cannot repair it')


def restore(root, apply=False):
    snapshot=json.loads((root/'outputs/e004_e005_input_verification.json').read_text(encoding='utf-8'))
    expected={name:digest for name,digest in snapshot['protected_sha256'].items()
              if Path(name).suffix in {'.md','.yaml','.json','.txt','.py'} and Path(name).name!='model.txt'}
    first=json.loads((root/'outputs/baselines/E004/F1/result.json').read_text(encoding='utf-8'))
    sources={name.replace('\\','/'):digest for name,digest in first['source_sha256'].items()}
    for name in TRAINING_SOURCES:
        path='src/stock_prediction/'+name
        expected[path]=sources[path]
    # Validate the entire set before writing any file. Large data/models stay outside this tool.
    changes=[]
    for name,digest in expected.items():
        path=root/name
        current=path.read_bytes()
        accepted=accepted_text_bytes(current,digest)
        if current!=accepted: changes.append((path,accepted))
    if apply:
        for path,accepted in changes: path.write_bytes(accepted)
    return dict(checked_files=len(expected),mode='restore' if apply else 'check',
                files_needing_restoration=[str(path.relative_to(root)) for path,_ in changes])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore',action='store_true',help='Write only LF/CRLF variants matching existing SHA-256')
    args=parser.parse_args()
    print(json.dumps(restore(Path.cwd(),args.restore),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
