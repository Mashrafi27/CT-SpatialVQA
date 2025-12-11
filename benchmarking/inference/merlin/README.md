# Merlin

- **GitHub**: https://github.com/StanfordMIMI/Merlin
- **Paper**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11230513/
- **Primary Tasks**: classification, retrieval, prediction, generation (radiology reports), segmentation.

## Repo Snapshot

- Upstream `StanfordMIMI/Merlin` is mirrored under `repo/` via git submodule; run `git submodule update --init --recursive` to sync.
- Pin a specific commit hash once the inference workflow is confirmed.

## Setup Checklist

- [ ] Sync `repo/` to the desired commit and note it here.
- [ ] Capture preprocessing requirements (e.g., DICOM to tensor pipelines) for the 10-sample sanity dataset.
- [ ] Implement or adapt inference script(s) for the tasks relevant to our spatial metrics.
- [ ] Export environment specs once HPC tests pass.

## Notes

The repo covers multiple modalities/tasks. Clearly mark which submodules we exercise for spatial reasoning benchmarks to avoid bloating dependencies.
