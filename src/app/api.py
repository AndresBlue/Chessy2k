"""Optional FastAPI surface for FEN / screenshot analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from src.chess_core.legal_validator import validate_fen
from src.engines.stockfish_client import StockfishClient

app = FastAPI(title="Chessy API", version="0.2.0")

STOCKFISH_PATH: str | None = None
CLASSIFIER_CHECKPOINT: str | None = None


@app.on_event("startup")
def startup_load_paths() -> None:
    """Resolve Stockfish and vision paths from env or repository defaults."""
    global STOCKFISH_PATH, CLASSIFIER_CHECKPOINT
    from src.app.runtime_paths import resolve_runtime_paths

    paths = resolve_runtime_paths()
    STOCKFISH_PATH = str(paths["stockfish"]) if paths.get("stockfish") else None
    CLASSIFIER_CHECKPOINT = (
        str(paths["classifier"]) if paths.get("classifier") else None
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "stockfish": bool(STOCKFISH_PATH),
        "classifier": bool(CLASSIFIER_CHECKPOINT),
    }


@app.post("/analyze/fen")
def analyze_fen(
    fen: str = Form(...),
    depth: int = Form(15),
    multipv: int = Form(5),
):
    if not STOCKFISH_PATH:
        return JSONResponse({"error": "Stockfish not configured"}, status_code=500)

    validation = validate_fen(fen)
    with StockfishClient(STOCKFISH_PATH) as engine:
        result = engine.analyze(fen, depth=depth, multipv=multipv)

    return {
        "validation": validation.to_dict(),
        "analysis": result.to_dict(),
    }


@app.post("/analyze/image")
async def analyze_image(
    image: UploadFile = File(...),
    side: str = Form("white"),
    depth: int = Form(15),
    multipv: int = Form(5),
):
    if not STOCKFISH_PATH:
        return JSONResponse({"error": "Stockfish not configured"}, status_code=500)

    import argparse

    from src.app.cli import run_image_pipeline

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    args = argparse.Namespace(
        image=tmp_path,
        fen=None,
        side=side,
        stockfish_path=STOCKFISH_PATH,
        reckless_path=None,
        depth=depth,
        movetime=None,
        multipv=multipv,
        threads=8,
        hash=1024,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        output_image=None,
        engine="stockfish",
        json=False,
    )

    try:
        result = run_image_pipeline(args)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return result


def create_app() -> FastAPI:
    return app
