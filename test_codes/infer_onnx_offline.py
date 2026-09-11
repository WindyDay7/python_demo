from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = REPO_ROOT / "artifacts" / "minirbt_intent_onnx_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline ONNX inference for the MiniRBT intent classifier."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help="Local ONNX bundle directory copied from the external training environment.",
    )
    parser.add_argument(
        "--model-file",
        default=None,
        help="Optional ONNX filename override. Defaults to manifest quantized model.",
    )
    parser.add_argument(
        "--text",
        action="append",
        help="Single input text. Pass multiple times for batch inference.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file with 'input' or 'text' field per line.",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top classes to return per sample.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional override for tokenizer max_length.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def read_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.extend(args.text)

    if args.input_jsonl is not None:
        with args.input_jsonl.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                text = row.get("input") or row.get("text")
                if text is None:
                    raise KeyError(
                        f"Line {line_number} is missing 'input' or 'text': {args.input_jsonl}"
                    )
                texts.append(str(text))

    normalized = [normalize_text(text) for text in texts if normalize_text(text)]
    if not normalized:
        raise ValueError("Provide at least one non-empty text via --text or --input-jsonl.")
    return normalized


def resolve_model_path(bundle_dir: Path, manifest: dict[str, Any], model_file: str | None) -> Path:
    if model_file is not None:
        candidate = Path(model_file)
        if candidate.is_absolute():
            return candidate
        return bundle_dir / candidate

    quantized = manifest.get("quantized_onnx_file")
    if quantized:
        return bundle_dir / str(quantized)
    return bundle_dir / str(manifest["onnx_file"])


def build_session_inputs(
    session: ort.InferenceSession,
    encoded: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    for ort_input in session.get_inputs():
        name = ort_input.name
        if name in encoded:
            inputs[name] = encoded[name]
            continue
        if name == "token_type_ids":
            inputs[name] = np.zeros_like(encoded["input_ids"], dtype=np.int64)
            continue
        raise KeyError(f"Tokenizer output is missing required ONNX input: {name}")
    return inputs


def top_k_predictions(
    probabilities: np.ndarray,
    id2label: dict[int, str],
    top_k: int,
) -> list[dict[str, Any]]:
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    return [
        {
            "label": id2label[int(index)],
            "score": float(probabilities[index]),
        }
        for index in top_indices
    ]


def main() -> None:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    mapping_path = bundle_dir / "label_mapping.json"

    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json is missing in bundle: {bundle_dir}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"label_mapping.json is missing in bundle: {bundle_dir}")

    manifest = load_json(manifest_path)
    mapping = load_json(mapping_path)
    id2label = {int(key): value for key, value in mapping["id2label"].items()}

    model_path = resolve_model_path(bundle_dir, manifest, args.model_file)
    max_length = args.max_length or int(manifest.get("max_length", 64))
    texts = read_texts(args)

    tokenizer = AutoTokenizer.from_pretrained(bundle_dir, local_files_only=True)
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="np",
    )

    session = ort.InferenceSession(
        model_path.as_posix(),
        providers=[args.provider],
    )
    session_inputs = build_session_inputs(session, encoded)
    logits = session.run(["logits"], session_inputs)[0]
    probabilities = softmax(logits)

    for text, sample_probabilities in zip(texts, probabilities):
        predictions = top_k_predictions(
            sample_probabilities,
            id2label,
            max(1, min(args.top_k, len(id2label))),
        )
        result = {
            "text": text,
            "prediction": predictions[0]["label"],
            "score": predictions[0]["score"],
            "top_k": predictions,
            "model_path": model_path.as_posix(),
        }
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()