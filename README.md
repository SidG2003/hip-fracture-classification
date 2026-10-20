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