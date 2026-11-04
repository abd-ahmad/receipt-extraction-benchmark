# Experiment Log

All results on the CORD validation split (100 receipts), field-level micro F1 over 5 summary fields.

| Version | Change | Precision | Recall | F1 | Fully correct |
|---|---|---|---|---|---|
| ocr_llm (v1) | EasyOCR (default) + qwen3:4b, integer schema, few-shot example | 0.719 | 0.579 | 0.642 | 25.0% |
| ocr_llm_v2 | All schema fields required (model previously closed JSON early, skipping later fields) | 0.710 | 0.612 | 0.657 | 27.0% |
| ocr_llm_v3 | v2 + snap predicted amounts to OCR numbers within edit distance 1 (grounding) | 0.724 | 0.624 | 0.670 | 29.0% |