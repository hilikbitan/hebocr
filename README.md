# Hebrew Hardcoded Subtitle OCR → SRT (Offline)

Production-ready MVP desktop tool that extracts burned-in Hebrew subtitles into UTF-8 SRT files.

## Features
- Automatic ROI detection with manual override.
- Smart frame sampling with perceptual hash gating.
- Deduplication using image similarity and text similarity.
- Timestamp smoothing (min duration, merge gaps, no overlaps).
- Offline OCR with EasyOCR (GPU-accelerated when available).
- Tkinter GUI with progress, logs, and cancel.

## Requirements
- Python 3.10+
- Windows 10/11
- FFmpeg (optional but recommended for faster frame extraction)
- NVIDIA GPU with CUDA for best performance (optional)

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### OCR Models
EasyOCR downloads language models on first run if missing. To keep runs fully offline, download the models once while online and cache them in the default EasyOCR cache directory.

## Usage
### GUI
```bash
python gui.py
```

### CLI
```bash
python extract.py <video_path> <output.srt> --config config.yaml
```

### Manual ROI
```bash
python extract.py <video_path> <output.srt> --manual-roi
```

## Configuration
Edit `config.yaml` to tune thresholds:
- `sample_fps`: frames per second to sample for OCR.
- `roi_scan_seconds`: duration used to auto-detect ROI.
- `search_vertical_band`: relative vertical band to search for subtitles.
- `phash_hamming_threshold`: perceptual hash change required to OCR.
- `min_ms_between_ocr`: minimum time between OCR calls.
- `text_similarity_threshold`: merging threshold for text similarity.
- `min_line_duration_ms`: enforced minimum subtitle duration.
- `max_merge_gap_ms`: maximum gap to merge subtitles.
- `min_confidence`: minimum OCR confidence to accept.
- `debug`: save debug artifacts and CSV logs.

## Debug Artifacts
When debug mode is enabled:
- Raw ROI crops and binarized crops are saved under `logs/session_<timestamp>/`.
- `decisions.csv` logs each OCR decision.

## Troubleshooting
- If you see a Windows error like `charmap codec can't encode character`, ensure you are launching via `gui.py` or `extract.py` which now force UTF-8 stdout/stderr.
- If EasyOCR is slow on CPU, ensure CUDA is installed and `torch.cuda.is_available()` returns true.
- If no subtitles are detected, try manual ROI selection or adjust `search_vertical_band` and `phash_hamming_threshold`.
- If FFmpeg is unavailable, OpenCV fallback will be used automatically.

## Project Structure
- `gui.py` — Tkinter app.
- `extract.py` — extraction pipeline.
- `train.py` — optional CRNN training.
- `model.py` — CRNN model definition.
- `roi.py` — ROI detection/selection helpers.
- `preprocess.py` — preprocessing pipeline.
- `dedup.py` — dedup and merge logic.
- `srt_writer.py` — SRT formatting.
- `ocr_backend.py` — OCR backend abstraction.
- `utils.py` — shared helpers.
