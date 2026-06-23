import subprocess, sys, os, tempfile

# ──────────────────────────────────────────────────────────────────────────────
# Runtime install of exact model-required versions.
# Done here (not requirements.txt) to avoid a huggingface-hub version conflict
# between transformers==4.57.1 (<1.0) and gradio 6.x (>=1.2.0) at build time.
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
from transformers import AutoModel, AutoTokenizer
from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse
import spaces

# ──────────────────────────────────────────────
# Model loading  (CPU at startup; ZeroGPU moves
# it to GPU per-request via @spaces.GPU)
# ──────────────────────────────────────────────
MODEL_NAME = "baidu/Unlimited-OCR"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
)
# Per ZeroGPU docs: models must be placed on cuda at module level.
# Outside @spaces.GPU, ZeroGPU runs a PyTorch CUDA emulation mode so
# this .cuda() call succeeds without a real GPU being present.
model = model.eval().cuda()
print("Model loaded on cuda (ZeroGPU emulation at startup).")

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = Server()


def pdf_to_images(pdf_path: str, dpi: int = 300):
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("pymupdf is required for PDF support: pip install pymupdf")
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


@app.api()
@spaces.GPU(duration=300)  # single-image OCR can take up to 5 min on long docs
def run_ocr(
    image_path: FileData,
    mode: str = "gundam",
    prompt: str = "document parsing.",
) -> dict:
    """
    Run OCR on a single image.

    mode: 'gundam' (crop, fast) or 'base' (full-res, accurate)
    Returns a dict with 'text' key containing the OCR result.
    """
    path = image_path["path"]
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")

    if mode == "gundam":
        base_size, image_size, crop_mode = 1024, 640, True
        ngram_window = 128
    else:
        base_size, image_size, crop_mode = 1024, 1024, False
        ngram_window = 128

    model.infer(
        tokenizer,
        prompt=f"<image>{prompt}",
        image_file=path,
        output_path=out_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=ngram_window,
        save_results=True,
    )

    # Collect text results (.txt / .md first, then anything readable)
    result_text = ""
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith(".txt") or fname.endswith(".md"):
            with open(os.path.join(out_dir, fname), "r", encoding="utf-8") as f:
                result_text += f.read() + "\n"

    if not result_text:
        for fname in sorted(os.listdir(out_dir)):
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        result_text += f.read() + "\n"
                except Exception:
                    pass

    return {"text": result_text.strip()}


@app.api()
@spaces.GPU(duration=600)  # multi-page / PDF parsing can take up to 10 min
def run_ocr_multi(
    image_paths: list,
    prompt: str = "Multi page parsing.",
) -> dict:
    """
    Run OCR on multiple images / PDF pages (base mode only).
    Returns a dict with 'text' key.
    """
    paths = [fp["path"] for fp in image_paths]
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")

    model.infer_multi(
        tokenizer,
        prompt=f"<image>{prompt}",
        image_files=paths,
        output_path=out_dir,
        image_size=1024,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=1024,
        save_results=True,
    )

    result_text = ""
    for fname in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    result_text += f.read() + "\n"
            except Exception:
                pass

    return {"text": result_text.strip()}


@app.get("/")
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


app.launch(show_error=True)
