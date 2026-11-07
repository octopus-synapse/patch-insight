"""Prepare and publish versioned Hub updates without logging credentials."""
import argparse
import hashlib
import json
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
    out=args.output
    if out.exists(): raise ValueError('use a fresh staging directory')
    out.mkdir(parents=True)
    snapshot_download(config['model_repo'],revision=config['model_revision'],local_dir=out/'model')
    shutil.rmtree(out/'model/.cache',ignore_errors=True)
    verify_bundle(out/'model')
    shutil.copytree(ROOT/'data',out/'dataset',ignore=shutil.ignore_patterns('v03'))
    for kind in ('model','dataset'):
        target=out/kind
        shutil.copytree(ROOT/'reports/v02',target/'evaluations/v02')
        shutil.copytree(ROOT/'docs',target/'project-docs',ignore=shutil.ignore_patterns('images'))
        card=target/'README.md'
        text=card.read_text().split('\n<!-- software-release -->')[0]
        text+=f'\n<!-- software-release -->\n## Software {args.tag}\n\nCheckpoint: {config["checkpoint_version"]}; weights unchanged in this release.\n'
        text+='New evaluation: 36 synthetic texts, authored and reviewed by the same AI; no human review.\nMacro-F1: heuristics 0.6910; frozen regression 0.6196; fine-tuned BERT 0.5584.\nSee [evaluation](evaluations/v02/test/README.md) and [protocol](project-docs/v02.md).\n'
        if kind=='dataset':
            text=text.replace('configs:\n','configs:\n- config_name: evaluation_v02\n  data_files:\n  - split: validation\n    path: v02/development.jsonl\n  - split: test\n    path: v02/test.jsonl\n',1)
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
