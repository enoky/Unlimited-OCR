import subprocess, sys, os, tempfile
from threading import Thread
from typing import Iterator

# ──────────────────────────────────────────────────────────────────────────────
# Runtime install of exact model-required versions.
# Done here (not requirements.txt) to avoid a huggingface-hub conflict between
# transformers==4.57.1 (<1.0) and gradio 6.x (>=1.2.0) at build time.
# ──────────────────────────────────────────────────────────────────────────────
_RUNTIME_PKGS = [
    "torch==2.10.0",
    "torchvision==0.25.0",
    "transformers==4.57.1",
]

print("Installing pinned runtime dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"] + _RUNTIME_PKGS,
    check=True,
)
print("Runtime deps installed.")

# ── Now safe to import ────────────────────────────────────────────────────────
import torch
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse
import spaces

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# Per ZeroGPU docs: place model on cuda at module level.
# ZeroGPU emulation mode lets .cuda() work at startup without a real GPU.
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "baidu/Unlimited-OCR"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
).eval().cuda()
print("Model ready.")

app = Server()


# ── PDF helper — CPU only ─────────────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """Convert every page of a PDF to a PNG. Returns list of file paths."""
    import fitz
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def _collect_output(out_dir: str) -> str:
    """Read all text/markdown files written by model.infer()."""
    result = ""
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith((".txt", ".md")):
            with open(os.path.join(out_dir, fname), "r", encoding="utf-8") as f:
                result += f.read() + "\n"
    if not result:
        for fname in sorted(os.listdir(out_dir)):
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        result += f.read() + "\n"
                except Exception:
                    pass
    return result.strip()


# ── Single-page OCR — streaming generator ────────────────────────────────────
#
# Gradio docs: any generator decorated with @app.api() automatically streams
# each yielded value to the client via SSE.  stream_every=0.1 means values
# are flushed at most every 100 ms.
#
# ZeroGPU: duration=60 → highest queue priority; one page per call.
#
@app.api(stream_every=0.1)
@spaces.GPU(duration=60)
def run_ocr(
    image_path: FileData,
    mode: str = "gundam",
    prompt: str = "document parsing.",
) -> Iterator[dict]:
    """
    Stream OCR output for one image page token-by-token.

    Yields dicts: {"text": str, "done": bool}

    mode: 'gundam' — fast (640 px crop)   ← ZeroGPU-friendly default
          'base'   — accurate (1024 px)
    """
    path    = image_path["path"]
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")

    if mode == "gundam":
        base_size, image_size, crop_mode, ngram_window = 1024, 640,  True,  128
    else:
        base_size, image_size, crop_mode, ngram_window = 1024, 1024, False, 128

    # ── Common infer kwargs ───────────────────────────────────────────────────
    _infer_kwargs = dict(
        prompt=f"<image>{prompt}",
        image_file=path,
        output_path=out_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=8192,
        no_repeat_ngram_size=35,
        ngram_window=ngram_window,
        save_results=True,
    )

    # ── Attempt real token streaming via TextIteratorStreamer ─────────────────
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    errors   = []

    def _infer_with_streamer():
        try:
            model.infer(tokenizer, **_infer_kwargs, streamer=streamer)
        except TypeError:
            # model.infer() doesn't accept `streamer` kwarg.
            # Re-run WITHOUT it so output files are actually written,
            # then signal the streamer loop to end.
            try:
                model.infer(tokenizer, **_infer_kwargs)
            except Exception as e:
                errors.append(str(e))
            finally:
                streamer.end()
        except Exception as e:
            errors.append(str(e))
            streamer.end()

    thread = Thread(target=_infer_with_streamer, daemon=True)
    thread.start()

    accumulated = ""
    for token in streamer:          # blocks until next token or end()
        accumulated += token
        yield {"text": accumulated, "done": False}

    thread.join()

    # ── Fallback: streamer not used → read file, stream word-by-word ──────────

    if not accumulated:
        full_text = _collect_output(out_dir)
        words     = full_text.split()
        acc       = ""
        for i, word in enumerate(words):
            acc += ("" if i == 0 else " ") + word
            if i % 5 == 0:                 # emit every 5 words
                yield {"text": acc, "done": False}
        yield {"text": full_text, "done": True}
    else:
        yield {"text": accumulated, "done": True}


# ── PDF explode — CPU only, no GPU ───────────────────────────────────────────
@app.api()
def explode_pdf(pdf_file: FileData) -> dict:
    """
    Convert a PDF into per-page image paths (CPU only — no GPU wasted on I/O).
    The frontend then calls run_ocr once per page, keeping each GPU slot to 60 s.
    """
    pages = pdf_to_images(pdf_file["path"], dpi=200)
    return {"pages": [{"path": p, "orig_name": os.path.basename(p)} for p in pages]}


# ── Static frontend ───────────────────────────────────────────────────────────
@app.get("/")
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


app.launch(show_error=True)
