"""Prepare and publish versioned Hub updates without logging credentials."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from patchinsight.model import verify_bundle, write_manifest


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--tag',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/release')
    args=parser.parse_args()
    version=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    if args.tag != 'v'+version: raise ValueError('tag and package version differ')
    config=json.loads((ROOT/'configs/release.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if args.publish and subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('commit reviewed changes before publication')
    out=args.output
    if out.exists(): raise ValueError('use a fresh staging directory')
    out.mkdir(parents=True)
    snapshot_download(config['model_repo'],revision=config['model_revision'],local_dir=out/'model')
    shutil.rmtree(out/'model/.cache',ignore_errors=True)
    metadata=verify_bundle(out/'model')
    if metadata['model_version'] != config['checkpoint_version']: raise ValueError('checkpoint version mismatch')
    shutil.copytree(ROOT/'data',out/'dataset',ignore=lambda directory,names: [n for n in names if re.fullmatch(r'v[0-9]+',n) and n not in config['dataset_versions']])
    evaluation=config['evaluation_version']
    evaluation_root=ROOT/'reports'/evaluation
    result_path=evaluation_root/'test/results.json'
    if not result_path.exists(): result_path=evaluation_root/'test.json'
    results=json.loads(result_path.read_text())['methods']
    for kind in ('model','dataset'):
        target=out/kind
        shutil.copytree(evaluation_root,target/'evaluations'/evaluation)
        shutil.copytree(ROOT/'docs',target/'project-docs',ignore=shutil.ignore_patterns('images'))
        card=target/'README.md'
        text=card.read_text().split('\n<!-- software-release -->')[0].replace('../docs/', 'project-docs/')
        text+=f'\n<!-- software-release -->\n## Software {args.tag}\n\nCheckpoint: {config["checkpoint_version"]}; See the pinned checkpoint revision in the release manifest.\n'
        text+='Synthetic evaluation authored and reviewed by the same AI; no human review.\n\n| Method | Macro-F1 |\n|---|---:|\n'
        for name,metrics in results.items():
            text+=f"| {name} | {metrics['macro_f1']:.4f} |\n"
        text+=f'\nSee evaluations/{evaluation} and project-docs/{evaluation}.md.\n'
        if kind=='dataset':
            extra=[]
            for dataset_version in config['dataset_versions']:
                extra.append(f'- config_name: evaluation_{dataset_version}')
                extra.append('  data_files:')
                for split,filename in [('train','train'),('validation','validation'),('calibration','calibration'),('test','test')]:
                    if not (target/dataset_version/(filename+'.jsonl')).exists():
                        if split=='validation': filename='development'
                        else: continue
                    extra.extend([f'  - split: {split}',f'    path: {dataset_version}/{filename}.jsonl'])
            text=text.replace('configs:\n','configs:\n'+'\n'.join(extra)+'\n',1)
        card.write_text(text)
        (target/'software-release.json').write_text(json.dumps({'tag':args.tag,'code_commit':commit,'checkpoint_version':config['checkpoint_version']},indent=2)+'\n')
    write_manifest(out/'model')
    verify_bundle(out/'model')
    result={'tag':args.tag,'code_commit':commit,'checkpoint_version':config['checkpoint_version'],'repositories':{}}
    if args.publish:
        api=HfApi()
        for kind in ('dataset','model'):
            repo=config[kind+'_repo']; repo_type=kind
            existing=[t.name for t in api.list_repo_refs(repo,repo_type=repo_type).tags]
            if args.tag in existing:
                from huggingface_hub import hf_hub_download
                release=json.loads(Path(hf_hub_download(repo,'software-release.json',revision=args.tag,repo_type=repo_type)).read_text())
                if release['code_commit']!=commit: raise ValueError('existing tag refers to another commit')
                revision=api.repo_info(repo,repo_type=repo_type,revision=args.tag).sha
            else:
                uploaded=api.upload_folder(repo_id=repo,repo_type=repo_type,folder_path=out/kind,commit_message=f'feat: publish software {args.tag} evaluation and documentation')
                revision=uploaded.oid
                api.create_tag(repo,tag=args.tag,revision=revision,repo_type=repo_type)
            result['repositories'][kind]={'repo':repo,'revision':revision}
    (out/'release-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
