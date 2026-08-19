# Unlimited-OCR — local

A clone of the [**baidu/Unlimited-OCR**](https://huggingface.co/spaces/baidu/Unlimited-OCR)
Hugging Face Space, altered to run **locally and fully offline** on a consumer
NVIDIA GPU.

The upstream Space targets ZeroGPU: it installs its own dependencies at startup,
schedules GPU time through the `spaces` package, and loads its frontend from
public CDNs. None of that applies on your own machine, so this fork drops the
ZeroGPU plumbing, pins the dependency set that the model actually needs, and
vendors every frontend asset. Once the model weights are cached, the app makes
no network requests at all.

The OCR model itself is unmodified — same weights, same `model.infer()`, same
results.

---

## What it does

Turns document images and PDFs into structured Markdown: headings, paragraphs,
tables (as HTML `<table>`), and per-region bounding boxes. Output streams into
the page token-by-token as it is generated.

<!-- Add a screenshot here: docs/screenshot.png -->

## Requirements

| | |
|---|---|
| GPU | NVIDIA, ~8 GB VRAM (the model occupies **6.8 GB** in bf16) |
| CUDA | A PyTorch build matching your GPU — see [Install](#install) |
| Python | 3.10+ |
| Disk | **6.3 GB** for model weights (cached in `~/.cache/huggingface`) |

Developed and tested on an **RTX 5080** (Blackwell, `sm_120`) with
`torch 2.13.0+cu130` on Windows 11 / Python 3.12. Roughly **7 seconds per page**
at 1200×700 in Long mode.

Nothing here is Windows-specific; the Linux steps are the same, minus the
`.venv\Scripts` path difference.

## Install

```bash
git clone https://github.com/enoky/Unlimited-OCR.git
cd Unlimited-OCR
python -m venv .venv
```

Activate it — `.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
on Linux/macOS — then install PyTorch for **your** GPU. Blackwell (RTX 50-series)
needs CUDA 12.8 or newer:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Older cards can use whichever CUDA build the
[PyTorch site](https://pytorch.org/get-started/locally/) recommends. Then:

```bash
pip install -r requirements.txt
```

pip will warn that `gradio 6.24.0 requires huggingface-hub>=1.16.0`. **Ignore
it** — that bound is metadata only, and gradio 6.24 runs correctly against the
0.36 that transformers requires. This is verified, not assumed.

## Run

```bash
python app.py
```

Open <http://127.0.0.1:7860>. First launch downloads ~6.3 GB of weights; after
that it is serving about **11 seconds** from cold.

| Variable | Default | Purpose |
|---|---|---|
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Set to `0.0.0.0` to expose on your LAN |
| `GRADIO_SERVER_PORT` | `7860` | Port to bind |
| `UNLIMITED_OCR_MODEL` | `baidu/Unlimited-OCR` | Model repo or a local directory |

## Using it

Drop in a PNG, JPG, WEBP, TIFF or PDF and press **START**.

- **Long** — 640 px crops, faster, the default. Handles dense multi-column pages.
- **Base** — single 1024 px view, slower, better on sparse or low-contrast scans.
- **Prompt** — defaults to `document parsing.`. Try `Free OCR.` for plain text
  with no layout markup, or `Parse the figure.` for charts.

PDFs are split server-side into per-page PNGs by PyMuPDF, then OCR'd one page at
a time so long documents stream progressively instead of blocking.

Two HTTP endpoints are exposed if you want to script it:

| Endpoint | Purpose |
|---|---|
| `/run_ocr` | One page. Streaming generator, yields `{text, done}` |
| `/explode_pdf` | PDF → per-page PNG paths (CPU only, no GPU) |

```python
from gradio_client import Client, handle_file

c = Client("http://127.0.0.1:7860")
print(c.predict(handle_file("page.png"), "gundam", "document parsing.",
                api_name="/run_ocr")["text"])
```

## ⚠️ Do not upgrade transformers

**`transformers` must stay at 4.57.1.** The model ships remote code written
before the transformers 5 rewrite. Under 5.x it loads without error and then
produces confident nonsense — transformers 5 rebuilds the tokenizer using its
declared `LlamaTokenizerFast` class, which installs SentencePiece processing
over what is really a ByteLevel BPE. Every space is eaten on the way in and the
way out, so the model is prompted with mangled text and its output is decoded
wrongly:

```
It is to do. It is to do. It is to do not to do not to do…
```

There is no warning and no traceback. A stray `pip install -U` on almost any
package will pull transformers 5 in, so if output ever degrades like this, check
the version first:

```bash
pip show transformers huggingface-hub
```

`requirements.txt` pins both, and `app.py` documents the reason at the point of
use.

## Changes from the upstream Space

`app.py` is a small diff — the OCR logic is untouched.

- **Removed the startup `pip install`.** The Space installed pinned versions at
  runtime to dodge a build-time dependency clash; dependencies now live in the
  venv, and `requirements.txt` carries the same pins.
- **`import spaces` is optional.** Falls back to a no-op `@GPU` decorator that
  runs the function on whatever GPU the process already owns, so the file still
  works unchanged on a real ZeroGPU Space.
- **Added a CUDA check** that fails with a clear message instead of a confusing
  crash inside the model, and prints the detected GPU.
- **Forced UTF-8 on stdout/stderr.** The model streams full-width CJK
  punctuation (`｜`); on a legacy Windows codepage that raises
  `UnicodeEncodeError` and kills the inference thread mid-generation.
- **`app.launch()` binds `127.0.0.1:7860`** and honours the environment
  variables above.
- **Vendored the frontend** (see below).

## Offline operation

`index.html` originally pulled `@gradio/client` and pdf.js from jsdelivr and its
fonts from Google Fonts. Everything is now checked into [`vendor/`](vendor/) and
served by `app.py` at `/vendor`:

| Asset | Files |
|---|---|
| `@gradio/client` 2.5.0 | 1 |
| pdf.js 4.4.168 + worker | 2 |
| pdf.js CMaps + standard fonts | 185 |
| JetBrains Mono + Inter (woff2) | 59 |

That is 5.4 MB on disk, but a page load fetches **669 KB across 11 requests**.
Everything else is lazy: font subsets are gated by `unicode-range` (6 of the 59
files load), the 1.4 MB pdf.js worker is pulled only when you open a PDF, and
the CMaps and standard fonts only for documents that actually reference them.

[`vendor/README.md`](vendor/README.md) records exact source URLs and versions for
refreshing them.

## Credits

- Model and original Space — [Baidu](https://huggingface.co/baidu/Unlimited-OCR).
  The model carries its own licence in the model repository.
- This fork only changes packaging and hosting.

> **Note:** the Hugging Face Space config front matter that used to head this
> file has been removed, since this repository's remote is GitHub and the front
> matter renders there as a stray table. If you ever deploy this to a Space,
> restore it from git history (`git show fece8f8:README.md`) and bump
> `sdk_version` to `6.24.0`.
