"""Run the frozen E004/E005 target comparison, preserving all prior experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .baselines import save_json
from .build_basic40 import sha256
from .e003 import run_fold


def verify_inputs(root):
    destination = root/'outputs/e004_e005_input_verification.json'
    if destination.exists():
        record = json.loads(destination.read_text(encoding='utf-8'))
        for name, digest in record['protected_sha256'].items():
            assert sha256(root/name) == digest, name
        for name, digest in record['input_sha256'].items():
            assert sha256(name) == digest, name
        return record
    manifest = json.loads((root/'outputs/full147_manifest.json').read_text(encoding='utf-8'))
    accepted = json.loads((root/'outputs/baselines/input_verification.json').read_text(encoding='utf-8'))
    prior = json.loads((root/'outputs/e003_input_verification.json').read_text(encoding='utf-8'))
    assert manifest['data_version'] == accepted['data_version']
    assert manifest['feature_set'] == 'Full147' and len(manifest['feature_names']) == 147
    inputs = {v['path']:v['sha256'] for v in manifest['datasets']['train']['files'].values()}
    inputs[str(root/'outputs/baselines/cache/labels.npy')] = accepted['labels_sha256']
    inputs[str(root/'data/raw/训练集.csv')] = accepted['raw_sha256']
    for name, digest in inputs.items():
        assert sha256(name) == digest, name
    assert (root/'evaluate.py').read_bytes() == (root/'reference/evaluate_official.py').read_bytes()
    assert sha256(root/'evaluate.py') == prior['official_script_sha256']
    protected = list((root/'config').glob('*.yaml'))
    protected += [root/'evaluate.py', root/'reference/evaluate_official.py',root/'docs/FEATURE_SPEC_V1.md']
    protected += [p for p in (root/'outputs').glob('*') if p.is_file()
                  and p.name != 'experiment_log.csv' and not p.name.startswith('e004_e005')]
    for experiment in ['E000','E001','E002','E003']:
        protected += [p for p in (root/'outputs/baselines'/experiment).glob('*/*') if p.is_file()]
    record = dict(baseline_commit='eb9660aec0b5147cce91cc1837be2406e9bf0300',
                  data_version=manifest['data_version'],test_labels_used=False,
                  input_sha256=inputs,protected_sha256={str(p.relative_to(root)):sha256(p) for p in protected},
                  original_log_text=(root/'outputs/experiment_log.csv').read_text(encoding='utf-8'))
    save_json(destination,record)
    print('Verified frozen inputs, original evaluator and E000-E003 artifacts',flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['E004','E005'])
    parser.add_argument('--fold',choices=['F1','F2','F3'])
    args = parser.parse_args()
    if bool(args.experiment) != bool(args.fold):
        parser.error('--experiment and --fold must be supplied together')
    root = Path.cwd()
    verify_inputs(root)
    if args.fold:
        run_fold(root,args.fold,args.experiment)
    else:
        for experiment in ['E004','E005']:
            for fold in ['F1','F2','F3']:
                subprocess.run([sys.executable,'-m','stock_prediction.e004_e005',
                                '--experiment',experiment,'--fold',fold],cwd=root,check=True)


if __name__ == '__main__':
    main()
