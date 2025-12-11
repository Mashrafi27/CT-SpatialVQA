# DeepTumorVQA

- **GitHub**: https://github.com/Schuture/DeepTumorVQA
- **Paper**: https://arxiv.org/pdf/2505.18915
- **Primary Tasks**: lesion/organ recognition, quantitative measurements (volume, HU), spatial reasoning, tumor staging.

## Repo Snapshot

- `repo/` folder tracks `Schuture/DeepTumorVQA` via submodule; sync with `git submodule update --init --recursive`.
- Lock commit hash once the initial sanity benchmark runs cleanly.

## Setup Checklist

- [ ] Configure the `repo/` submodule (commit hash, necessary branches) or extract inference scripts specific to VQA evaluation.
- [ ] Define prompts/questions aligned with the paper's four assessment buckets (recognition, measurement, reasoning, medical decision-making).
- [ ] Execute the 10-case sanity set, ensuring volumetric measurements are supported.
- [ ] Export the requirements file from the stabilized environment.

## Notes

Because this benchmark targets the same spatial skills we care about, prioritize automating metric computation (e.g., comparing predicted counts vs. ground truth) once inference runs.
