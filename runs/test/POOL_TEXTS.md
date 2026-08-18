# Pool texts

Rebuild locally (not shipped — ~86 MB):

```powershell
$env:PYTHONPATH="scripts"
python "scripts/build_pool_texts.py" `
  --input-run runs/test/r_output_trec_rag_2026_phase1_doc.tsv `
  --output runs/test/pool_texts_phase1_doc.json `
  --depth 100 --overwrite
```
