import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.ml
def test_tiny_local_multilabel_roundtrip(tmp_path):
    """Random tiny model verifies plumbing; never reported as a real experiment."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from patchinsight import LABELS
    from patchinsight.experiment import save_json, tokenized_dataset
    from patchinsight.model import BertPredictor, write_manifest

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nCorrigi\na\nconsulta\n.\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.save_pretrained(tmp_path)
    model = BertForSequenceClassification(BertConfig(
        vocab_size=9, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, num_labels=5, problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={k:i for i,k in enumerate(LABELS)}))
    row = {"text":"Corrigi a consulta.", "labels":dict(zip(LABELS, [True, False, False, True, False]))}
    dataset = tokenized_dataset([row], tokenizer, 16)
    batch = {k:v.unsqueeze(0) for k,v in dataset[0].items()}
    before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    model.save_pretrained(tmp_path, safe_serialization=True)
    save_json(tmp_path / "patchinsight.json", {"trained":True,"labels":list(LABELS),
              "model_version":"test-random-tiny", "thresholds":[.5]*5, "max_length":16})
    write_manifest(tmp_path)
    predictor = BertPredictor(tmp_path)
    scores, states, truncated, tokens = predictor.predict("Corrigi a consulta.")
    assert not truncated
    assert tokens > 0
    assert all(0 <= score <= 1 for score in scores.values())
    assert set(states) == set(LABELS)
    _, states, truncated, _ = predictor.predict("consulta " * 50)
    assert truncated
    assert "absent" not in states.values()
    vocab.write_text("tampered")
    with pytest.raises(ValueError):
        BertPredictor(tmp_path)


@pytest.mark.real_model
def test_real_checkpoint():
    from patchinsight.model import BertPredictor
    directory = os.environ.get("PATCHINSIGHT_MODEL")
    if not directory:
        pytest.skip("PATCHINSIGHT_MODEL must point to the real trained checkpoint")
    predictor = BertPredictor(Path(directory))
    assert predictor.metadata["experiment"]["base_model"] == "neuralmind/bert-base-portuguese-cased"
    assert predictor.metadata["selected_epoch"] >= 1
    scores, _, _, _ = predictor.predict("Revisei as consultas com SQL e reduzi a espera em 20%.")
    assert len(scores) == 5
    examples = ["Revisei as consultas com SQL.", "OAuth 2.0"]
    batched = predictor.predict_many(examples)
    for text, result in zip(examples, batched):
        single = predictor.predict(text)
        assert result[1:] == single[1:]
        assert list(result[0].values()) == pytest.approx(list(single[0].values()), abs=1e-5)
