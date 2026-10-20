# Hip-fracture classification

Research pilot: distinguish **two expert-confirmed fracture classes** with a
frozen SKELEX encoder and a trainable linear head. This is not a fracture detector,
does not handle normal hips or the other fracture classes, and is not validated
for clinical use or treatment decisions.

## Setup

Use Python 3.11. Direct dependency versions are pinned to the stack used for the
initial SKELEX loading smoke test.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Images, manifests, model weights, credentials, and training runs must stay out of
Git. Their usual locations and formats are ignored.

## Pilot assumptions

- Exactly two expert-confirmed fracture classes; pass their actual names through
  `--classes`. `type_a` and `type_b` below are placeholders, not clinical labels.
- **One image per unique patient across the whole dataset.** The data provider
  must verify this. No patient ID, study ID, view, or label-status field is required.
- Each selected image must show enough information to support its assigned label.
- Training, validation, and test images must be independent. An image from any
  other fracture category or a normal hip is outside this pilot's scope.
- Document population/exclusion criteria with your expert before interpreting
  results. A successful two-class pilot does not establish six-class performance.

## Input data

Put images and a CSV manifest under ignored `data/`. The manifest has exactly two
columns, in this order:

```csv
image_path,label
images/example_001.png,type_a
images/example_002.png,type_b
```

Image paths are relative to the manifest, or absolute. Accepts single-frame 8-bit
PNG/JPEG/BMP/TIFF images, converted to RGB and processed using the pretrained
model's resize/normalization configuration (224 x 224 for the downloaded SKELEX
checkpoint). DICOM and high-bit-depth images require a separate, clinically
reviewed export/windowing step; they are not silently converted. Inspect the
resized images with your expert to ensure fracture detail remains visible.

Validation rejects invalid labels, missing/corrupt images, and byte-identical
duplicate files. It cannot identify the same patient, near-duplicates,
re-encoded copies, or site/view-specific shortcuts.

```bash
hip-fracture validate \
  --manifest data/manifest.csv \
  --classes type_a type_b
```

## Reproducible splits

### Chest-versus-pelvis anatomy trial

For the Mendeley dataset "A Multi-Class X-ray Image Dataset for Human Anatomy
Classification", place the extracted folders under ignored `Anatomy Dataset/`.
Prepare only Chest and Pelvis, retaining one copy of each byte-identical image:

```bash
python scripts/prepare_anatomy.py --dataset "Anatomy Dataset" --output data/anatomy
hip-fracture split --manifest data/anatomy/manifest.csv \
  --classes Chest Pelvis --output data/anatomy/splits --seed 2025
```

The preparation script never changes the original images. It writes a validated
two-column manifest, `excluded_duplicates.csv` mapping excluded copies to retained
images, and `preparation.json` with counts and limitations. Conflicting labels for
identical images fail explicitly. Exact deduplication cannot establish patient
independence or remove re-encoded/near-duplicate images.

Use these manifests and `--classes Chest Pelvis` with the training command below,
using a separate run such as `runs/anatomy-001`. This is an anatomy-classification
engineering check, not evidence of fracture-subtype performance. Check the
dataset's original terms and citation before redistributing data or results.

If your teammate supplies independent train/validation/test manifests, use them
directly. Otherwise, create class-stratified image-level splits:

```bash
hip-fracture split \
  --manifest data/manifest.csv \
  --classes type_a type_b \
  --output data/splits \
  --seed 2025
```

Defaults are approximately 60% train, 20% validation, 20% test, subject to
rounding. Every split must contain both classes; inadequate class counts fail
explicitly. These fractions are engineering defaults, not a sample-size
recommendation. Generated manifests contain absolute image paths, so regenerate
or update them if you move the dataset. Existing output directories are never
overwritten.

## Frozen-encoder training

Use the downloaded SKELEX directory from the adjacent checkout; no new download
or copying of the research scripts is needed:

```bash
hip-fracture train \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --classes type_a type_b \
  --model ../SKELEX/models/SKELEX \
  --output runs/pilot-001 \
  --epochs 20 \
  --batch-size 4 \
  --lr 0.001 \
  --device cpu
```

The encoder:
- Loads all pretrained encoder weights and rejects missing/mismatched weights.
  Decoder-only unused-weight notices from Transformers are expected.
- Disables MAE masking and random patch shuffling, retaining all image patches.
- Stays frozen and in evaluation mode, including during head training.
- Supplies its CLS-token feature vector to a new two-output linear head.

Only **2,050 parameters** are trained for the downloaded 1,024-dimensional SKELEX
encoder. The reconstruction decoder is not used. No augmentation or encoder
fine-tuning is included in this first baseline.

`--device auto` selects CUDA, then Apple MPS, then CPU if available. Requesting an
unavailable device fails rather than falling back silently. Local validation is
on CPU; GPU/MPS execution and cross-platform determinism are not certified.
The encoder is recomputed each batch; large datasets may need GPU resources or a
later feature-caching step. Benchmark before committing to long runs.

Each new run directory contains:
- `config.json`: settings, class counts, actual device, package versions, backbone
  weight hash, and the patient-independence assumption.
- `history.json`: per-epoch training loss and validation metrics.
- `best.pt`: the head from the epoch with lowest validation cross-entropy,
  ordered class names, backbone path/hash/config, preprocessing, and hashes of
  training/validation images. The frozen backbone is not duplicated.

Checkpoints are for prediction/evaluation; optimizer-state resume is not
implemented. Seeds make splits and CPU tests repeatable, but do not guarantee
identical results across hardware/library versions.

## Held-out evaluation

Only evaluate the reserved test set once model choices are finalized. Do not use
test performance to choose training settings.

```bash
hip-fracture evaluate \
  --checkpoint runs/pilot-001/best.pt \
  --manifest data/splits/test.csv \
  --output runs/pilot-001/test \
  --device cpu
```

Writes `metrics.json` and `predictions.csv`. Reports per-class precision, recall,
F1 and support; macro-F1; accuracy/balanced accuracy; AUROC treating class 1 as
positive; and a confusion matrix with explicit class order. Prediction CSV
probability columns follow that class order. Rejects byte-identical overlap with
training/validation data. No confidence intervals or external validation are
implemented yet.

## Single-image prediction

```bash
hip-fracture predict \
  --checkpoint runs/pilot-001/best.pt \
  --image data/images/example_001.png \
  --device cpu
```

Returns the predicted pilot class and both probabilities. These are not
calibrated clinical confidence scores. The model always chooses one of the two
classes and has **no normal/other/uncertain rejection mechanism**.

Evaluation/prediction require the original backbone directory. If you move it,
pass `--model /new/path/to/SKELEX`; weight SHA-256, encoder settings, and
preprocessing must match the training checkpoint.

## Tests

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
```

Tests generate temporary synthetic images and tiny randomly initialized MAE
checkpoints. They cover manifest validation, duplicate detection, split
reproducibility/isolation, freezing, full patch visibility, head updates,
training, checkpoint selection/reloading, known metric values, inference, and
overlap/tampering rejection. No patient data or SKELEX download is needed.
GitHub Actions runs the same offline tests after installing dependencies.

These tests establish software behavior, not medical accuracy.

## Upstream model and licensing

This project uses independently written training code and the Transformers
implementation of `ViTMAEModel`. No SKELEX research scripts are vendored.

- Model: https://huggingface.co/skhoha/SKELEX
- Research code: https://github.com/skhoha/SKELEX
- Initial downloaded revision: `368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e`
- Initial weight SHA-256:
  `81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8`

Both the upstream code and model metadata specify **CC BY-NC-ND 4.0**.
This project does not grant additional rights to those materials. Obtain
clarification from the authors before distributing adaptations/fine-tuned
weights or considering commercial use. A private repository does not remove
license obligations.