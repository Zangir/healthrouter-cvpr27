# HealthRouter-Bench: a routing benchmark over frozen medical image encoders

> This folder is the HealthRouter-Bench data release. It is hosted next to the code during the
> anonymous review period and moves to a dataset hub on publication.

**Status: under review.** This repository accompanies an anonymous CVPR 2027 submission. During review the files are public under the terms below.

## Summary

HealthRouter-Bench is the evaluation data of a per-sample model-routing study. One frozen pool of ten image encoders (four generic, one biomedical, five in-domain) is evaluated on four pooled medical tasks, four external cohorts under five acquisition conditions, and two non-medical tasks. Each encoder has a linear head per task. Twenty routers and ensembles (HealthRouter, its confidence-gated harness, and 18 baselines from six families) choose, for each image, which encoders to run.

This repository releases the **derived artefacts** for every source whose terms allow redistributing them:
- split definitions;
- cross-fitted per-encoder predictions;
- router features;
- every method's test outputs;
- the frozen evaluation protocol.

**No images are included.** Images are keyed by the sha256 of their source bytes and by their row index. No file name, patient identifier or source metadata is included. Sources whose terms do not clearly allow redistribution are rebuilt from your own download with the companion code.

## Contents

| Path | What it is |
|---|---|
| `data/splits/{pcam,cbis_ddsm,eurosat}_labels.csv` | `row, image_sha256, label` for each in-source pool |
| `data/splits/{pcam,cbis_ddsm,eurosat}_splits.csv` | `row, image_sha256, seed, outer_fold, role`. Role is `train_oof` (inner training part, used out-of-fold), `validation` or `test`. 3 split seeds × 5 outer folds |
| `data/predictions/{tag}_crossfit.npz` | per seed and fold: row indices and class probabilities of all 10 encoders for the training part (out-of-fold, 5-fold cross-fitting), validation and test parts; chosen head regularisation `C` |
| `data/features/{tag}_router_features_convnext_t.npz` | router features (ImageNet ConvNeXt-T, 768-d) for every row |
| `data/router_outputs/{tag}_test_outputs.npz` | for every method (HealthRouter, harness, 18 baselines, ablations, Oracle): test-set probabilities and the boolean mask of encoders run, per seed |
| `data/external/external_{camelyon17_ood,idrid,cmmd}.npz` (+ `_labels.csv`) | external cohorts. For each condition (clean, grayscale, blur, low resolution, JPEG): encoder probabilities, router features, and every method's outputs and run masks |
| `protocol/router_plan.json` | the frozen evaluation plan and its four timestamped amendments |
| `protocol/router_*.json` | the aggregate results reported in the paper (all tasks, including those whose per-image files are not released) |
| `protocol/candidate_pool.json` | encoders, hub ids, revisions, licences, measured cost, head and input protocol |
| `tools/match_images.py` | joins released rows to images in your own download by sha256 |
| `build_log.json` | order checks performed when the files were built |

Encoder order in every array: `imagenet_convnext, dinov2_base, dinov2_large, siglip2_base, biomedclip, rad_dino, phikon, phikon_v2, retfound, mammo_clip`.

## Sources: released or rebuilt

| Task | Source | Rows | Released here? |
|---|---|---|---|
| Pathology (in-source) | PatchCamelyon official test, balanced subsample | 2,000 | yes (CC0 1.0) |
| Mammography (in-source) | CBIS-DDSM test, exact-deduplicated against train | 614 | yes (CC BY 3.0, TCIA) |
| Non-medical | EuroSAT RGB, 200 per class | 2,000 | yes (MIT) |
| External pathology | Camelyon17-WILDS OOD test hospital, balanced sample | 1,000 | yes (CC0 1.0) |
| External fundus | IDRiD disease grading | 516 | yes (CC BY 4.0) |
| External mammography | CMMD, MLO views (TOMPEI-CMMD join) | 396 | yes (CC BY 4.0) |
| Chest (in-source) | Shenzhen tuberculosis set | 662 | **no**: rebuild (no licence stated by the host) |
| Fundus (in-source) | APTOS 2019 holdout | 562 | **no**: rebuild (competition terms) |
| External chest | Montgomery County set | 138 | **no**: rebuild (no licence stated by the host) |
| Non-medical | DTD, 50 per class | 2,350 | **no**: rebuild ("for research purposes", no licence) |

## How to build or obtain the data

1. **Released artefacts.** After access is approved, download this repository. With the companion code installed, `python scripts/release_reproduce.py --bench <this folder>` recomputes 499 numbers in the paper (pathology and mammography columns of the main table, the EuroSAT column, the single-encoder table and the per-cohort external results). It then compares them with `protocol/`. All 499 match.
2. **Images.** Download each source from its host under its terms, then join by hash: `python tools/match_images.py --labels data/splits/cbis_ddsm_labels.csv --folder <your CBIS-DDSM images>`. For PatchCamelyon use `--pcam-h5`, and for Hugging Face parquet mirrors use `--parquet`.
3. **Withheld sources and macro scores.** Every macro score pools all four medical tasks, and HealthRouter retrieves neighbours across the pooled training set. Reproducing them needs the Shenzhen and APTOS artefacts, which you rebuild from your own downloads with the companion code: feature extraction, `router_candidates.py`, `router_experiment.py --phase final` and the aggregation scripts. The splits are deterministic functions of the seeds in `protocol/router_plan.json`.

## Splits

Each in-source task is split into five class-stratified outer folds, repeated over three seeds (0, 1, 2), so every row is a test row once per seed. Within each outer training part, 20% is held out for validation. Five-fold cross-fitting on the rest gives the out-of-fold predictions that every router, gate and combiner trains on. Every method selected its configuration on validation, and test folds were scored once. External cohorts are test-only. Their models were trained on the in-source pools with one 80/20 train/validation split.

**Duplicates.** Some sources contain byte-identical images within the released rows. These are kept as in the source and reported here:

| Source | Duplicated hashes | With conflicting labels |
|---|---|---|
| PatchCamelyon subsample | 33 | 0 |
| CBIS-DDSM | 4 | 2 |
| IDRiD | 6 | 3 |

Use `row`, not the hash alone, as the key.

## Licence and terms

- The derived files are released under **CC BY-NC 4.0**. They include outputs of RETFound (CC BY-NC 4.0) and Mammo-CLIP (CC BY-NC-SA 4.0). Files derived from Mammo-CLIP outputs remain subject to Mammo-CLIP's own share-alike terms in addition.
- Source terms continue to apply. Cite and attribute each source:
  - PatchCamelyon and Camelyon17: CC0 1.0.
  - CBIS-DDSM: CC BY 3.0; cite Sawyer-Lee et al., TCIA 2016, doi:10.7937/K9/TCIA.2016.7O02S9CY, and follow the TCIA Data Usage Policy.
  - IDRiD: CC BY 4.0.
  - CMMD: CC BY 4.0, TCIA.
  - EuroSAT: MIT.
- No patient may be re-identified. No images are redistributed here.
- By using these files you agree to: use them for non-commercial research only; make no attempt to re-identify any patient; not redistribute the files; obtain each source dataset from its own host under its own terms; and cite the HealthRouter paper.

## Citation

Anonymous CVPR 2027 submission, "HealthRouter: Routing Each Image to the Right Collection of Expert Models". Full citation on publication.

## Contact

Contact details will be added after the anonymous review period.
