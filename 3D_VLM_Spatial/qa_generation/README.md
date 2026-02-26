# QA Generation v2 (sanitized)

This folder keeps a clean, repeatable split for generating new QA pairs from
`3D_VLM_Spatial/validation_reports_output.json` without reusing cases that
already have QA.

## Prepare case list

Example (exclude existing QA from previous runs):

```bash
python 3D_VLM_Spatial/qa_generation_v2/prepare_case_split.py \
  --reports 3D_VLM_Spatial/validation_reports_output.json \
  --existing 3D_VLM_Spatial/spatial_qa_output.json \
  --existing 3D_VLM_Spatial/spatial_qa_filtered.json \
  --existing 3D_VLM_Spatial/spatial_qa_processed.jsonl \
  --existing 3D_VLM_Spatial/spatial_qa_m3d.jsonl \
  --existing 3D_VLM_Spatial/spatial_qa_radfm.jsonl \
  --existing 3D_VLM_Spatial/spatial_qa_ctchat.jsonl \
  --existing 3D_VLM_Spatial/spatial_qa_resampled_merlin.jsonl \
  --target-total 1000 \
  --seed 13
```

Outputs:
- `selected_cases.json` — list of case_ids to use (existing + sampled).
- `existing_case_ids.json` — cases already covered.
- `remaining_case_ids.json` — candidate pool before sampling.
- `manifest.json` — summary with counts and inputs.

## How to use

- Point your QA generation code to `selected_cases.json` and only generate for
  the **sampled** cases (you can diff against `existing_case_ids.json`).
- Or, if your generator can skip existing cases, just use `selected_cases.json`.

Keep any new generation scripts in this folder so outputs stay isolated.
