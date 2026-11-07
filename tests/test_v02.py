import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from patchinsight import LABELS
from patchinsight.cli import batch_records
from patchinsight.demo import make_server
from patchinsight.service import Analyzer


def test_batch_preserves_order_errors_and_limits():
    analyzer = Analyzer('heuristic')
    stream = io.StringIO('{"id":"a","text":"Revisei com SQL."}\nnot json\n{"text":" "}\n{"id":"b","text":"OAuth 2.0"}\n')
    rows = list(batch_records(stream, analyzer, batch_size=2))
    assert [r['line'] for r in rows] == [1, 2, 3, 4]
    assert rows[0]['id'] == 'a' and rows[3]['id'] == 'b'
    assert 'error' in rows[1] and 'error' in rows[2]
    assert rows[3]['prediction']['components']['measurement']['state'] == 'absent'
    with pytest.raises(ValueError, match='limite de itens'):
        list(batch_records(io.StringIO('{"text":"x"}\n'*2), analyzer, max_items=1))
    with pytest.raises(ValueError, match='100000'):
        list(batch_records(io.StringIO('x'*100001), analyzer))


def test_action_method_requires_known_present_components():
    from patchinsight.rules import structural_rules
    states = dict.fromkeys(LABELS, 'absent')
    states.update(action='present', method='unknown')
    assert 'Action–Method' not in structural_rules('Revisei.', states)[0]
    states['method'] = 'present'
    assert 'Action–Method' in structural_rules('Revisei por comparação manual.', states)[0]


def test_loopback_demo_inference_and_origin_protection():
    server = make_server(Analyzer('heuristic'), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base) as response:
            assert response.status == 200
            assert 'Descrição profissional'.encode() in response.read()
        request = Request(base+'/v1/analyze', data=json.dumps({'text':'Corrigi com SQL.'}).encode(),
                          headers={'Content-Type':'application/json'})
        with urlopen(request) as response:
            parsed = json.load(response)
            assert parsed['backend'] == 'heuristic'
            assert 'Action–Method' in parsed['structures']
        request.add_header('Origin', 'https://untrusted.example')
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        invalid = Request(base+'/v1/analyze', data=b'{"text":" "}',
                          headers={'Content-Type':'application/json'})
        with pytest.raises(HTTPError) as error:
            urlopen(invalid)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_suite_freeze_and_provenance():
    pytest.importorskip('sklearn')
    from pathlib import Path
    from patchinsight.research import audit, suite_rows
    root = Path(__file__).resolve().parents[1] / 'data/v02'
    for split in ('development', 'test'):
        rows, _ = suite_rows(root, split)
        result, _ = audit(rows)
        assert result['n'] == 36 and result['origins'] == 34


def test_abstention_never_completes_structure_from_unknown():
    class Predictor:
        version = 'fixture'

        def predict_many(self, texts):
            return [({k: .51 for k in LABELS}, dict.fromkeys(LABELS, 'present'), False, 4)
                    for _ in texts]
    pytest.importorskip('sklearn')
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.backend = 'bert'
    analyzer.predictor = Predictor()
    analyzer.policy = {'temperature': 1, 'confidence_cutoff': .9, 'policy_id': 'fixture'}
    prediction = analyzer.analyze('Revisei por comparação manual.')
    assert all(c.state == 'unknown' and c.score is None for c in prediction.components.values())
    assert prediction.structures == []
    assert 'review_method' in prediction.diagnostics


def test_policy_rejects_other_method_or_bundle(tmp_path, monkeypatch):
    pytest.importorskip('sklearn')
    from patchinsight.data import sha256
    import patchinsight.model

    class Predictor:
        metadata = {'thresholds': [.5]*5}

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(patchinsight.model, 'BertPredictor', Predictor)
    (tmp_path/'manifest.json').write_text('{}')
    policy = {'method':'bert_finetuned', 'model_manifest_sha256':sha256(tmp_path/'manifest.json'),
              'labels':list(LABELS), 'thresholds':[.5]*5, 'temperature':1,
              'confidence_cutoff':None, 'policy_id':'fixture'}
    path = tmp_path/'policy.json'
    for key, value, message in [('method','frozen_logistic','outro método'),
                                ('model_manifest_sha256','wrong','outro bundle')]:
        path.write_text(json.dumps(dict(policy, **{key:value})))
        with pytest.raises(ValueError, match=message):
            Analyzer('bert', tmp_path, policy=path)


def test_v03_data_is_frozen_and_disjoint():
    import hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'data/v03'
    manifest=json.loads((root/'freeze.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    seen_texts=set(); seen_groups=set()
    for split,expected in [('train',96),('validation',48),('calibration',32),('test',64)]:
        rows=[json.loads(s) for s in (root/f'{split}.jsonl').read_text().splitlines()]
        assert len(rows)==expected
        texts={r['text'] for r in rows}; groups={r['group_id'] for r in rows}
        assert not seen_texts & texts and len(texts)==len(rows)
        assert not seen_groups & groups
        seen_texts.update(texts);seen_groups.update(groups)
