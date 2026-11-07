"""Controlled training and selection. Test predictions are computed only after selection is frozen."""
import json,sys,shutil,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from patchinsight import LABELS
from patchinsight.experiment import fine_tune,targets,save_json,environment
from patchinsight.model import BertPredictor,write_manifest
from patchinsight.assessment import extended_metrics,temperature_scale,binary_log_loss
from patchinsight.metrics import metrics

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/refinement-v03')
    parser.add_argument('--reports',type=Path,default=ROOT/'reports/v03')
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    reports=args.reports;reports.mkdir(parents=True,exist_ok=True)
    if (out/'selected-model').exists(): raise ValueError('already selected; test cannot be reused for tuning')
    freeze=json.loads((ROOT/'data/v03/freeze.json').read_text())
    for filename,digest in freeze['sha256'].items():
     assert hashlib.sha256((ROOT/'data/v03'/filename).read_bytes()).hexdigest()==digest
    all_rows=[json.loads(s) for split in ['train','validation','calibration','test'] for s in (ROOT/'data/v03'/f'{split}.jsonl').read_text().splitlines()]
    assert len({r['text'] for r in all_rows})==len(all_rows)
    def load(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
    def score(model,rows):
     return np.concatenate([model.score_batch([r['text'] for r in rows[i:i+8]]) for i in range(0,len(rows),8)])
    def evaluate(y,p,t):
     return extended_metrics(y,p,thresholds=t)
    
    train=load(ROOT/'data/splits/train.jsonl')
    aug=[]
    for r in train:
     text=r['text'].replace('Abordagem: ','Usando ').replace('Escopo: ','Volume de ')
     if text!=r['text']: aug.append(dict(r,text=text))
    train+=aug+load(ROOT/'data/v03/train.jsonl')*2+load(ROOT/'data/v02/development.jsonl')*2
    (out/'training-inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train))
    val=load(ROOT/'data/v03/validation.jsonl');cal=load(ROOT/'data/v03/calibration.jsonl')
    y=np.asarray(targets(val));yc=np.asarray(targets(cal))
    config=json.loads((ROOT/'configs/experiment.json').read_text());config['epochs']=6;config['selection']='validation_macro_f1_then_log_loss'
    candidates=[]
    for name,lr,init in [('threshold_only',None,None),('continued',1e-5,str(ROOT/'artifacts/run-42/model')),('base',2e-5,None)]:
     run=out/name
     if name=='threshold_only': directory=ROOT/'artifacts/run-42/model'
     else:
      directory=run/'model';cfg=dict(config,learning_rate=lr)
      if init:cfg['initial_checkpoint']=init
      run.mkdir(exist_ok=True)
      if not (run/'training.json').exists():
       fine_tune(cfg,train,val,run,args.device,{'dataset_v03':json.loads((ROOT/'data/v03/freeze.json').read_text()),'augmentation':'marker rewriting; train only; natural rows oversampled twice'})
     model=BertPredictor(directory,args.device);pv=score(model,val);pc=score(model,cal)
     temperatures=np.geomspace(.25,4,81)
     temp=float(min(temperatures,key=lambda t:binary_log_loss(yc,temperature_scale(pc,t))))
     pv=temperature_scale(pv,temp)
     thresholds=[]
     for i in range(5):
      choices=np.round(np.arange(.1,.901,.05),2)
      def objective(t):
       pred=pv[:,i]>=t;tp=int(((y[:,i]==1)&pred).sum());fp=int(((y[:,i]==0)&pred).sum());fn=int(((y[:,i]==1)&~pred).sum())
       return (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,-abs(float(t)-.5))
      thresholds.append(float(max(choices,key=objective)))
     report=evaluate(y,pv,thresholds)
     candidates.append({'name':name,'directory':str(directory),'temperature':temp,'thresholds':thresholds,'validation':report})
     save_json(reports/'development.json',{'candidates':candidates,'train_effective_rows':len(train),'environment':environment()})
     del model
     import torch,gc
     gc.collect();torch.cuda.empty_cache()
    best=max(candidates,key=lambda c:(c['validation']['macro_f1'],-c['validation']['mean_binary_brier']))
    save_json(reports/'selection.json',best)
    selected=out/'selected-model'
    if selected.exists(): raise ValueError('selected bundle already exists; do not repeat test')
    shutil.copytree(best['directory'],selected)
    meta=json.loads((selected/'patchinsight.json').read_text())
    meta.update(model_version='0.3.0',thresholds=best['thresholds'],calibration={'temperature':best['temperature'],'source':'v03_calibration'},refinement=best['name'])
    save_json(selected/'patchinsight.json',meta);write_manifest(selected)
    # Compare every model on the same held-out texts after all choices are frozen.
    test=load(ROOT/'data/v03/test.jsonl');yt=np.asarray(targets(test))
    result={}
    for name,directory,temp,t in [('original',ROOT/'artifacts/run-42/model',1,[.5]*5),('refined',best['directory'],best['temperature'],best['thresholds'])]:
     model=BertPredictor(directory,args.device);p=temperature_scale(score(model,test),temp)
     result[name]=evaluate(yt,p,t)
     save_json(reports/(name+'-test-predictions.json'),{'ids':[r['id'] for r in test],'targets':yt.tolist(),'scores':p.tolist(),'thresholds':t})
     old=load(ROOT/'data/v02/test.jsonl');po=temperature_scale(score(model,old),temp)
     result[name+'_v02_diagnostic']=evaluate(np.asarray(targets(old)),po,t)
     del model;gc.collect();torch.cuda.empty_cache()
    save_json(reports/'test.json',{'methods':result,'selection':best,'reference':'synthetic_same_ai_reviewed','environment':environment()})
    print(json.dumps({n:{k:m[k] for k in ['macro_f1','macro_average_precision','mean_binary_brier','mean_binary_log_loss']} for n,m in result.items()}))


if __name__ == "__main__":
    main()
