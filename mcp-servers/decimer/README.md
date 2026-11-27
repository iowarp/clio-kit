# DECIMER MCP

Read a chemical structure drawing (a PNG or JPEG of one molecule, for example cropped
from a paper's figure) into SMILES with [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer).
RDKit then checks the SMILES. The result is a **model's reading** of the image. Have the
person compare it with the drawing, for example in Ketcher, before using it.

DECIMER is the most accurate of the four optical structure recognition models that
ChemGraph benchmarks: exact match 0.899 over 722 images, against 0.835 for MolNexTR,
0.824 for MolScribe and 0.766 for OCSRGlyph. TensorFlow is installed only in this
server's environment. The agent's own Python never imports it.

## Tools

| Tool | What it does |
| --- | --- |
| `image_to_smiles(image_path, hand_drawn=false)` | Returns DECIMER's SMILES and RDKit's check of it: `valid`, canonical SMILES, formula, molecular weight (g/mol), heavy atoms, stereo centres and InChIKey. It also returns the decoder's per-token confidence (mean, lowest, and the five lowest tokens) and the timing. An invalid SMILES is still returned as the model wrote it, with `rdkit.valid: false` and RDKit's reason. `hand_drawn=true` uses DECIMER's model for sketches. |
| `crop_image(image_path, box, unit="pixel", output_dir?)` | Cuts `[left, top, right, bottom]` (pixels, or fractions with `unit="fraction"`) out of a figure. The crop is saved as PNG under `<output_dir or $CLIO_KIT_ARTIFACTS or cwd>/chemistry/decimer/`, ready for `image_to_smiles`. |

Errors end with `[reason]`: `image_not_found`, `unreadable_image` (the file's content
is not an image, whatever its extension, or it cannot be decoded), `image_too_large`
(over 20 MB), `bad_box`, `model_download_failed`, `model_failed`.

## Model weights

The first call downloads the weights from Zenodo and later calls reuse them:

- the printed-structure model is 285 MiB (299 MB) zipped;
- the hand-drawn model is 285 MiB (299 MB) zipped. Its tokenizer comes from the printed
  model's archive, so the hand-drawn model's first use needs that archive too.

Both models together take 664 MB unpacked. They are cached under the user cache folder
(`platformdirs`, `~/.cache/decimer-mcp` on Linux), or under `$DECIMER_MCP_CACHE`. The
download is reported in the result text of the call that made it.

Measured on 4 CPU cores:

| Step | Time |
| --- | --- |
| First call (download and load) | 129 to 132 s |
| Load from cache in a new process | about 32 s |
| Reading one image | 1.4 to 6 s |

## Install

`uv sync` in this folder. The pins are `decimer==2.8.1` and `tensorflow-cpu==2.20.0` on
Linux (`tensorflow==2.20.0` elsewhere), with Python 3.11 or 3.12 (`.python-version`
3.12). DECIMER requires `tensorflow`. On Linux a uv override replaces it with
`tensorflow-cpu`, which provides the same module without the GPU build. The environment
is about 1.9 GB.

## Tests

```bash
uv run --project mcp-servers/decimer pytest mcp-servers/decimer -m "not integration"
DECIMER_MCP_DOWNLOAD=1 uv run --project mcp-servers/decimer pytest mcp-servers/decimer -m integration
```

The offline tests check the images, the crops, the RDKit check and the tool texts, with
the model replaced by a stub. The integration test draws caffeine and aspirin with RDKit
and checks that DECIMER reads both back exactly (same canonical SMILES). It runs when the
weights are cached, or when `DECIMER_MCP_DOWNLOAD=1` allows the download.

**This test is the easy case.** A clean RDKit depiction is close to DECIMER's training
images. Scanned or photographed figures, structures with abbreviations (Ph, OTs, R
groups), overlapping labels, arrows or text near the drawing, and Markush structures are
harder. Expect misreadings there. That is why every result asks for a check against the
image.

## NOTICE

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0):

- the image checks come from `tools/ocsr_core.py`;
- the cached, once-loaded DECIMER call and its cold-start reporting come from
  `tools/ocsr_backends.py`;
- the model choice and benchmark figures come from `tools/ocsr_registry.json`.

DECIMER is by Kohulan Rajan, Achim Zielesny and Christoph Steinbeck (MIT licence). Its
weights are published on Zenodo (records 8300489 and 10781330).
