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

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# Per ZeroGPU docs: place model on cuda at module level.
# ZeroGPU runs PyTorch CUDA emulation outside @spaces.GPU, so .cuda() works
# at startup without a real GPU. Moving to cuda here is more efficient than
# doing it lazily inside @spaces.GPU (CUDA transfers are optimised at startup).
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

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
app = Server()


# ── PDF helper (CPU-only, no GPU needed) ──────────────────────────────────────
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


# ── Single-page OCR — the only GPU endpoint ───────────────────────────────────
# ZeroGPU tip: shorter duration = higher queue priority.
# Process ONE page per call so each GPU slot is brief and shared fairly.
# For PDFs the frontend calls this endpoint once per page and streams results.
@app.api()
@spaces.GPU(duration=60)          # 60 s default = highest queue priority
def run_ocr(
    image_path: FileData,
    mode: str = "gundam",         # gundam (640 px crop) is faster; base = 1024 px
    prompt: str = "document parsing.",
) -> dict:
    """
    OCR one image page.

    mode: 'gundam' — fast (image_size=640, crop_mode=True)  ← default, ZeroGPU-friendly
          'base'   — accurate (image_size=1024, crop_mode=False)
    Returns {'text': str}.
    """
    path = image_path["path"]
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")

    if mode == "gundam":
        base_size, image_size, crop_mode, ngram_window = 1024, 640, True, 128
    else:
        base_size, image_size, crop_mode, ngram_window = 1024, 1024, False, 128

    model.infer(
        tokenizer,
        prompt=f"<image>{prompt}",
        image_file=path,
        output_path=out_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=8192,           # per-page cap — keeps GPU slot short
        no_repeat_ngram_size=35,
        ngram_window=ngram_window,
        save_results=True,
    )

    result_text = ""
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith((".txt", ".md")):
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


# ── PDF explode — CPU only, no GPU ───────────────────────────────────────────
@app.api()
def explode_pdf(pdf_file: FileData) -> dict:
    """
    Convert a PDF to a list of per-page image paths.
    Returns {'pages': [FileData, ...]} so the frontend can call
    run_ocr once per page, keeping each GPU slot short.
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
