# Vendored frontend dependencies

`index.html` used to load these from cdn.jsdelivr.net, fonts.googleapis.com and
fonts.gstatic.com. They are checked in here and served locally by `app.py` at
`/vendor`, so the app makes no external requests and works fully offline.

| Path | Package | Version | Source |
|---|---|---|---|
| `gradio-client.min.js` | `@gradio/client` | 2.5.0 | `https://cdn.jsdelivr.net/npm/@gradio/client@2.5.0/dist/index.min.js` |
| `pdf.min.mjs` | `pdfjs-dist` | 4.4.168 | `https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.min.mjs` |
| `pdf.worker.min.mjs` | `pdfjs-dist` | 4.4.168 | `https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.min.mjs` |
| `pdfjs/standard_fonts/` (16 files) | `pdfjs-dist` | 4.4.168 | `https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/standard_fonts/` |
| `pdfjs/cmaps/` (169 files) | `pdfjs-dist` | 4.4.168 | `https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/cmaps/` |
| `fonts.css` + `fonts/` (59 files) | Google Fonts | JetBrains Mono, Inter | `https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap` |

Total on disk is about 4 MB, but a page load fetches only a small fraction of
it — every one of these is resolved lazily.

## Fonts

`fonts.css` is the Google stylesheet with each `fonts.gstatic.com` URL rewritten
to `/vendor/fonts/`. All subsets are vendored (latin, latin-ext, cyrillic,
cyrillic-ext, greek, greek-ext, vietnamese), but every `@font-face` keeps its
`unicode-range`, so a browser downloads only what the page renders — in practice
the 6 latin files, ~250 KB.

## pdf.js assets

`standard_fonts/` supplies the 14 standard PostScript faces (Helvetica, Times,
Courier, Symbol, ZapfDingbats) for PDFs that reference them without embedding.
`cmaps/` supplies the predefined CJK character maps (UniJIS, UniGB, UniKS,
UniCNS, …) in pdf.js's packed `.bcmap` format.

`index.html` points pdf.js at both when it calls `getDocument`:

```js
const PDFJS_ASSETS = {
  cMapUrl: "/vendor/pdfjs/cmaps/",
  cMapPacked: true,
  standardFontDataUrl: "/vendor/pdfjs/standard_fonts/",
};
```

Without them, a preview of a PDF relying on non-embedded standard fonts or CJK
encodings falls back to substituted glyphs. This affects the in-page preview
only — OCR rasterises server-side in `explode_pdf` via PyMuPDF, so accuracy is
never touched by these files.

Note that `standard_fonts/` usually stays untouched on a desktop machine: pdf.js
defaults to `useSystemFonts: true` and substitutes a locally installed face
(Arial for Helvetica, and so on) rather than downloading anything. The vendored
data is the fallback for machines that lack those fonts. Loading the same PDF
with `useSystemFonts: false` forces the download and is how to verify the wiring
— it fetches `LiberationSans-Regular.ttf`. CMaps have no such shortcut and are
always fetched when a document uses a predefined CJK encoding.

## Refreshing

Re-download from the URL above and update the version in this table. `app.py`
registers the MIME types for `.js`, `.mjs`, `.css`, `.woff2`, `.bcmap` and
`.pfb` explicitly, because Python infers them from the Windows registry, where
several are wrong or missing.
