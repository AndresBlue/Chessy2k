"""CLI for Chessy: screenshot or FEN to recommended move."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.chess_core.legal_validator import validate_fen
from src.engines.stockfish_client import StockfishClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chessy: screenshot/FEN to best move recommendation",
    )
    parser.add_argument("--image", type=str, help="Path to board screenshot PNG")
    parser.add_argument("--fen", type=str, help="FEN string (manual input)")
    parser.add_argument(
        "--side",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Side to move when inferring from screenshot",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish binary (default: engines/stockfish/...)",
    )
    parser.add_argument(
        "--reckless-path",
        type=str,
        default=None,
        help="Path to Reckless binary (for --engine reckless)",
    )
    parser.add_argument("--depth", type=int, default=15, help="Search depth")
    parser.add_argument("--movetime", type=int, default=None, help="Search time in ms")
    parser.add_argument("--multipv", type=int, default=5, help="Number of top lines")
    parser.add_argument("--threads", type=int, default=8, help="Engine threads")
    parser.add_argument("--hash", type=int, default=1024, help="Engine hash MB")
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help="Path to piece classifier checkpoint",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default=None,
        help="Path to save annotated output image",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["stockfish", "reckless"],
        default="stockfish",
        help="Engine to use for move recommendation",
    )
    return parser


def _resolve_stockfish_path(args: argparse.Namespace) -> str:
    if args.stockfish_path:
        return args.stockfish_path
    from src.app.runtime_paths import resolve_runtime_paths

    path = resolve_runtime_paths().get("stockfish")
    if path is None:
        raise SystemExit(
            "Stockfish binary not found. Pass --stockfish-path or place it under "
            "engines/stockfish/"
        )
    return str(path)


def _resolve_reckless_path(args: argparse.Namespace) -> str:
    if args.reckless_path:
        return args.reckless_path
    from src.app.runtime_paths import resolve_runtime_paths

    path = resolve_runtime_paths().get("reckless")
    if path is None:
        raise SystemExit(
            "Reckless binary not found. Pass --reckless-path or place it under "
            "engines/reckless/"
        )
    return str(path)


def _analyze_with_uci(args: argparse.Namespace, fen: str) -> dict:
    if args.engine == "reckless":
        from src.engines.reckless_client import RecklessClient

        client: StockfishClient = RecklessClient(
            path=_resolve_reckless_path(args),
            threads=args.threads,
            hash_mb=args.hash,
        )
    else:
        client = StockfishClient(
            path=_resolve_stockfish_path(args),
            threads=args.threads,
            hash_mb=args.hash,
        )
    with client as engine:
        analysis = engine.analyze(
            fen=fen,
            depth=args.depth if args.movetime is None else None,
            movetime_ms=args.movetime,
            multipv=args.multipv,
        )
    return {
        "best_move_uci": analysis.best_move.uci,
        "best_move_san": analysis.best_move.san,
        "evaluation": analysis.best_move.score_str,
        "principal_variation": analysis.best_move.pv,
        "top_moves": [
            {"uci": m.uci, "san": m.san, "score": m.score_str, "pv": m.pv}
            for m in analysis.top_moves
        ],
        "engine_time_ms": analysis.time_ms,
        "engine": args.engine,
    }


def run_fen_baseline(args: argparse.Namespace) -> dict:
    """FEN → validate → engine analysis."""
    t0 = time.perf_counter()
    validation = validate_fen(args.fen)
    if not validation.is_valid:
        return {
            "error": "Invalid FEN",
            "validation": validation.to_dict(),
            "total_time_ms": (time.perf_counter() - t0) * 1000,
        }
    engine_result = _analyze_with_uci(args, validation.fen)
    return {
        "fen": validation.fen,
        "validation": validation.to_dict(),
        **engine_result,
        "total_time_ms": (time.perf_counter() - t0) * 1000,
    }


def run_image_pipeline(args: argparse.Namespace) -> dict:
    """Screenshot → FEN → validate → engine."""
    from src.chess_core.state_tracker import GameStateTracker
    from src.vision.pipeline import VisionPipeline
    from src.vision.visualizer import draw_best_move_arrow

    t0 = time.perf_counter()

    checkpoint = args.classifier_checkpoint
    if not checkpoint:
        from src.app.runtime_paths import resolve_runtime_paths

        ckpt = resolve_runtime_paths().get("classifier")
        if ckpt is None:
            raise SystemExit(
                "Vision checkpoint not found. Pass --classifier-checkpoint or place "
                "data/checkpoints/vision/best.pt"
            )
        checkpoint = str(ckpt)

    pipeline = VisionPipeline(checkpoint_path=checkpoint)
    vision_result = pipeline.process(args.image, side=args.side)

    tracker = GameStateTracker()
    view_orientation = args.side
    tracker_result = tracker.update_from_vision(
        board_matrix=vision_result.board_matrix,
        orientation=view_orientation,
        side_hint=args.side,
    )

    fen = tracker_result.fen
    validation = validate_fen(fen)
    engine_result = _analyze_with_uci(args, fen)

    output_image_path = args.output_image
    if output_image_path is None and args.image:
        output_image_path = str(
            Path(args.image).with_stem(Path(args.image).stem + "_annotated")
        )

    if output_image_path and vision_result.debug_image is not None:
        annotated = draw_best_move_arrow(
            vision_result.debug_image,
            engine_result["best_move_uci"],
            orientation=view_orientation,
        )
        import cv2

        cv2.imwrite(output_image_path, annotated)
        engine_result["output_image"] = output_image_path

    return {
        "fen": fen,
        "fen_pieces": vision_result.fen_pieces,
        "orientation": view_orientation,
        "confidence_per_square": vision_result.confidence.tolist(),
        "ambiguous_squares": vision_result.ambiguous_squares,
        "vision_time_ms": vision_result.time_ms,
        "tracker": tracker_result.to_dict(),
        "validation": validation.to_dict(),
        **engine_result,
        "total_time_ms": (time.perf_counter() - t0) * 1000,
    }


def print_result(result: dict) -> None:
    """Pretty-print analysis result."""
    if "error" in result:
        print(f"ERROR: {result['error']}")
        if "validation" in result:
            for err in result["validation"].get("errors", []):
                print(f"  - {err}")
        return

    print(f"FEN: {result.get('fen', 'N/A')}")
    if result.get("orientation"):
        print(f"Orientation: {result['orientation']}")
    if result.get("validation"):
        v = result["validation"]
        status = "VALID" if v["is_valid"] else "INVALID"
        print(f"Validation: {status} ({v['legal_move_count']} legal moves)")
        for w in v.get("warnings", []):
            print(f"  Warning: {w}")
    print(f"\nBest move: {result.get('best_move_san')} ({result.get('best_move_uci')})")
    print(f"Evaluation: {result.get('evaluation')}")
    if result.get("principal_variation"):
        print(f"PV: {' '.join(result['principal_variation'])}")
    print("\nTop moves:")
    for i, m in enumerate(result.get("top_moves", [])[:5], 1):
        print(f"  {i}. {m.get('san', m.get('uci'))} ({m.get('uci')}) — {m.get('score', '')}")
    if result.get("ambiguous_squares"):
        print(f"\nAmbiguous squares: {result['ambiguous_squares']}")
    if result.get("output_image"):
        print(f"\nAnnotated image: {result['output_image']}")
    print(f"\nTotal time: {result.get('total_time_ms', 0):.1f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.image:
        result = run_image_pipeline(args)
    elif args.fen:
        result = run_fen_baseline(args)
    else:
        parser.error("Provide --image or --fen")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)

    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
