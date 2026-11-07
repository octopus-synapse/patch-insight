"""Verify saved selection, evaluate fixed references, and describe the local v0.3 candidate."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'src')
from patchinsight.assessment import extended_metrics,paired_group_bootstrap,temperature_scale
from patchinsight.research import Methods
from patchinsight.model import BertPredictor,write_manifest,verify_bundle
from patchinsight.schema import Prediction
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--reports',type=Path,default=Path('reports/v03'))
parser.add_argument('--run',type=Path,default=Path('artifacts/refinement-v03'))
args=parser.parse_args()
def main():
    root=args.reports
    r=json.loads((root/'test.json').read_text());sel=json.loads((root/'selection.json').read_text())
    rows=[json.loads(s) for s in Path('data/v03/test.jsonl').read_text().splitlines()]
    y=np.asarray([[int(row['labels'][k]) for k in ('action','context','method','result','measurement')] for row in rows])
    a=json.loads((root/'refined-test-predictions.json').read_text());b=json.loads((root/'original-test-predictions.json').read_text())
    r['uncertainty']=paired_group_bootstrap(y,(np.asarray(a['scores'])>=sel['thresholds']).astype(int),(np.asarray(b['scores'])>=.5).astype(int),[x['group_id'] for x in rows])
    methods=Methods(Path('artifacts/run-42'),'cuda')
    for name in ['heuristic','frozen_logistic']:
     p=np.concatenate([methods.score([row['text'] for row in rows[i:i+8]],name) for i in range(0,len(rows),8)])
     r['methods'][name]=extended_metrics(y,p)
    for c in json.loads((root/'development.json').read_text())['candidates']:
     if c['name']==sel['name']:continue
     model=BertPredictor(Path('artifacts/run-42/model') if c['name']=='threshold_only' else c['directory'],'cuda')
     p=temperature_scale(np.concatenate([model.score_batch([x['text'] for x in rows[i:i+8]]) for i in range(0,len(rows),8)]),c['temperature'])
     r['methods'][c['name']]=extended_metrics(y,p,c['thresholds'])
     del model
    r['target_met']=r['methods']['refined']['macro_f1']>=r['methods']['original']['macro_f1']+.1 and all(r['methods']['refined'][m]<=r['methods']['original'][m] for m in ['mean_binary_brier','mean_binary_log_loss'])
    (root/'test.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    bundle=args.run/'selected-model'
    selected_metrics=r['methods']['refined']
    (bundle/'README.md').write_text(f"""# PatchInsight checkpoint 0.3.0 — local experimental candidate

Selected candidate: {sel['name']}. Temperature: {sel['temperature']}.
Component thresholds: {sel['thresholds']}. Software/schema: 0.2.0.
New test macro-F1: {selected_metrics['macro_f1']:.6f}.

Synthetic controlled compositions, same authoring/reviewing AI, no human
validation. Shared patterns make this benchmark artificially easy. The known
v0.2 diagnostic is not an independent test. No real-resume generalization or
professional competency evaluation has been established. Calibration remains
experimental; the bundle applies its temperature once during inference.

Training and selection were local. See reports/v03 and docs/v03.md in the code
repository for data, comparisons, limitations and reproduction. MIT code and
checkpoint with NeuralMind notice preserved; CC BY 4.0 synthetic data.
This candidate has not been uploaded as a public model release.
""")
    write_manifest(bundle);verify_bundle(bundle)
    model=BertPredictor(bundle,'cuda')
    p=model.score_batch([x['text'] for x in rows])
    assert np.allclose(p,a['scores'],atol=1e-5)
    for m in ['mean_binary_brier','mean_binary_log_loss']:
     assert abs(extended_metrics(y,p)[m]-r['methods']['refined'][m])<1e-5
    print(json.dumps({'target_met':r['target_met'],'selection':sel['name'],'bootstrap':r['uncertainty']}))


if __name__ == "__main__":
    main()
