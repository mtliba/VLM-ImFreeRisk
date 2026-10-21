# Data

The `*.jsonl` files in this directory are the exact splits used in the paper. Each line is one question:

```json
{"sample_id": "vqarad_test_0", "dataset": "vqa_rad", "image_id": "...", "image_path": "images/<file>",
 "question": "...", "answer": "...", "answer_type": "CLOSED|OPEN", "question_type": "PRES", "modality": null, "location": null}
```

`image_path` is relative to `data/<benchmark>/`. Images are **not** redistributed; obtain them from the original sources and place them under `data/<benchmark>/images/`, or run the preparation scripts, which download them and regenerate these files identically:

| Benchmark | Source | Script | Splits used |
|---|---|---|---|
| VQA-RAD | Lau et al., *Scientific Data* 2018 — `flaviagiammarino/vqa-rad` on the Hugging Face Hub (question types from the original OSF release) | `scripts/download_data.py` | official test (451 QA / 203 images); 15 % of official-train images (seed 0) held out as validation (272 QA); 1,521 QA for training |
| SLAKE (English) | Liu et al., ISBI 2021 — `BoKelvin/SLAKE` | `scripts/download_data.py` | official train / validation / test (4,919 / 1,053 / 1,061) |
| PathVQA | He et al., 2020 — `flaviagiammarino/path-vqa` | `scripts/prepare_benchmarks.py` | image-disjoint subsample 5,000 / 1,000 / 2,000 of the official splits (seed 0) |
| PMC-VQA | Zhang et al., 2023 — `RadGenome/PMC-VQA` | `scripts/prepare_benchmarks.py` | official 2,000-item clean test set; image-disjoint 5,000 / 1,000 train / validation subsample |
| OmniMedVQA | Hu et al., CVPR 2024 — `foreverbeliever/OmniMedVQA` | `scripts/prepare_benchmarks.py` | modality-stratified, image-disjoint 5,000 / 1,000 / 2,000 subsample (eight modalities) |

For the multiple-choice benchmarks the `question` field already contains the options (`Options: A. ... B. ...`) and `answer` is the option letter; `scripts/common.py` scores them by letter.

Images are disjoint across splits for every benchmark. Each dataset remains under its original licence.
