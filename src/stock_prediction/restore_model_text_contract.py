"""Check/restore only accepted LF/CRLF text bytes for the E006/E007 stage."""
import argparse
import json
from pathlib import Path

from .restore_text_contract import accepted_text_bytes


def restore(root,apply=False):
    snapshot=json.loads((root/'outputs/e006_e007_input_verification.json').read_text(encoding='utf-8'))
    expected={name:digest for name,digest in snapshot['protected_sha256'].items()
              if Path(name).suffix in {'.md','.yaml','.json','.txt','.py'} and Path(name).name!='model.txt'}
    first=json.loads((root/'outputs/baselines/E006/F1/result.json').read_text(encoding='utf-8'))
    for filename in ['ranking.py','e006_e007.py']:
        path='src/stock_prediction/'+filename
        expected[path]=first['source_sha256'][path]
    changes=[]
    for name,digest in expected.items():
        path=root/name; current=path.read_bytes(); accepted=accepted_text_bytes(current,digest)
        if current!=accepted: changes.append((path,accepted))
    if apply:
        for path,accepted in changes: path.write_bytes(accepted)
    return dict(checked_files=len(expected),mode='restore' if apply else 'check',
                files_needing_restoration=[str(path.relative_to(root)) for path,_ in changes])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--restore',action='store_true')
    args=parser.parse_args()
    print(json.dumps(restore(Path.cwd(),args.restore),ensure_ascii=False,indent=2))
