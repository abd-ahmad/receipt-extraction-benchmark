# Experiment Log

## Development (validation split, 100 receipts)

Field-level micro F1 over 5 summary fields.

| Version | Change | Precision | Recall | F1 | Fully correct |
|---|---|---|---|---|---|
| ocr_llm (v1) | EasyOCR (default) + qwen3:4b, integer schema, few-shot example | 0.719 | 0.579 | 0.642 | 25.0% |
| ocr_llm_v2 | All schema fields required (model previously closed JSON early, skipping later fields) | 0.710 | 0.612 | 0.657 | 27.0% |
| ocr_llm_v3 | v2 + snap predicted amounts to OCR numbers within edit distance 1 (grounding) | 0.724 | 0.624 | 0.670 | 29.0% |
| ocr_llm_v4 | v3, but never snap to a shorter number (v3: 13 fixed / 9 broken, all 6 shortenings were harmful) | 0.741 | 0.638 | 0.686 | 30.0% |
| donut | naver-clova-ix/donut-base-finetuned-cord-v2, no changes (trained on CORD train split) | 0.983 | 0.991 | 0.987 | 95.0% |

**Final choice for approach A: v4.** Snapping fixes 16 values and breaks 7 (mostly small change amounts).
Further rule tuning was stopped to avoid overfitting to the 100 validation receipts.

## Final results (test split, 100 receipts)

Both methods were frozen before evaluating on the test split.

| Method | Precision | Recall | F1 | Fully correct | Seconds/receipt |
|---|---|---|---|---|---|
| OCR + LLM (v4) | 0.683 | 0.679 | 0.681 | 35.0% | 9.39 (OCR + LLM) |
| Donut | 0.955 | 0.972 | 0.963 | 85.0% | 3.42 |