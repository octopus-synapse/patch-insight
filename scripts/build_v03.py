"""Deterministic grouped refinement data, reviewed by the authoring AI."""
import hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LABELS=('action','context','method','result','measurement')
root=ROOT/'data/v03'
rows=[]
order=list(range(60)); random.Random(314).shuffle(order)
splits={i:('train' if n<24 else 'validation' if n<36 else 'calibration' if n<44 else 'test') for n,i in enumerate(order)}
for i,line in enumerate((root/'source/scenarios.tsv').read_text().splitlines()):
 c,a,m,r,q=line.split('|'); split=splits[i]
 variants=[('11111',f'{c}. {a} usando {m}; {r[0].lower()+r[1:]}. Volume de {q}.'),
           ('10101',f'{a} por meio de {m}, abrangendo {q}.'),
           ('01010',f'{c}; {r[0].lower()+r[1:]}.')]
 bits,text=[('00101',f'Pretendo investigar usando {m}, considerando {q}.'),
            ('00000','OAuth 2.0, Python 3.11 e Java 21.'),
            ('10000',f'{a}.'),('00011',f'{r}. Volume de {q}.'),
            ('00100',f'Procedimento: {m}.'),('00001',f'{q}.'),
            ('00000', 'Não realizei a investigação sobre '+a.split(' ',1)[1]+'.')][i%7]
 # The last negative variant has no performed contribution; no result asserted.
 variants.append((bits,text))
 for j,(bits,text) in enumerate(variants):
  rows.append(dict(id=f'v03-{i:03}-{j}',group_id=f'v03-{i:03}',split=split,text=text,
   labels=dict(zip(LABELS,[b=='1' for b in bits])),synthetic=True,human_reviewed=False,
   reviewer='same_authoring_ai',review_note='Cláusulas revisadas: investigação não implica resultado; efeito explícito é resultado; meio e quantidade preservados em futuro.',
   source_ref=f'source/scenarios.tsv:{i+1}',license='CC-BY-4.0'))
# A repeated version-list negative stays only once per split, preserving unique evaluation rows.
# Use scenario-specific identifiers as irrelevant numbers rather than repeated text.
for row in rows:
 if row['text']=='OAuth 2.0, Python 3.11 e Java 21.':
  row['text']+=' Referência '+row['group_id']+'.'
assert len(rows) == 240
assert len({r['text'] for r in rows}) == len(rows), 'duplicate texts across origins'
for split in ['train','validation','calibration','test']:
 content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['split']==split)
 path=root/f'{split}.jsonl'
 if (root/'freeze.json').exists() and path.read_text()!=content: raise ValueError('already frozen')
 path.write_text(content)
files=['source/scenarios.tsv','README.md',*[f'{s}.jsonl' for s in ['train','validation','calibration','test']]]
manifest={'seed':314,'counts':{s:sum(r['split']==s for r in rows) for s in splits.values()},'sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}}
if (root/'freeze.json').exists() and json.loads((root/'freeze.json').read_text()) != manifest: raise ValueError('frozen protocol changed')
(root/'freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest['counts'])
