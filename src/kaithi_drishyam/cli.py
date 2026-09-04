"""Command-line interface for Kaithi-Drishyam."""

import argparse
import sys
from pathlib import Path

from loguru import logger


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="kaithi-drishyam",
        description="OCR system for digitizing historical Kaithi script documents",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a document image")
    process_parser.add_argument("image", type=Path, help="Path to document image")
    process_parser.add_argument(
        "-o", "--output", type=Path, help="Output file path (default: stdout)"
    )
    process_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    server_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the CRNN model")
    train_parser.add_argument(
        "--data-dir", type=Path, required=True, help="Path to training data directory"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model accuracy")
    eval_parser.add_argument(
        "--test-dir", type=Path, required=True, help="Path to test data directory"
    )
    eval_parser.add_argument(
        "--model", type=Path, required=True, help="Path to model checkpoint"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "process":
        return _process_document(args)
    elif args.command == "server":
        return _start_server(args)
    elif args.command == "train":
        return _train_model(args)
    elif args.command == "evaluate":
        return _evaluate_model(args)

    return 0


def _process_document(args: argparse.Namespace) -> int:
    """Process a single document image."""
    logger.info(f"Processing document: {args.image}")
    # TODO: Implement document processing pipeline
    logger.warning("Document processing not yet implemented")
    return 0


def _start_server(args: argparse.Namespace) -> int:
    """Start the API server."""
    import uvicorn

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "kaithi_drishyam.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _train_model(args: argparse.Namespace) -> int:
    """Train the CRNN model."""
    logger.info(f"Training model with data from: {args.data_dir}")
    # TODO: Implement training pipeline
    logger.warning("Model training not yet implemented")
    return 0


def _evaluate_model(args: argparse.Namespace) -> int:
    """Evaluate model accuracy."""
    logger.info(f"Evaluating model: {args.model}")
    # TODO: Implement evaluation pipeline
    logger.warning("Model evaluation not yet implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
