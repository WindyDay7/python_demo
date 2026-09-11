from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "minirbt_intent_training" / "best_model"
DEFAULT_BUNDLE_DIR = REPO_ROOT / "artifacts" / "minirbt_intent_onnx_bundle"


class SequenceClassifierExportWrapper(torch.nn.Module):
    def __init__(self, base_model: AutoModelForSequenceClassification) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_inputs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.base_model(**model_inputs)
        return outputs.logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fine-tuned MiniRBT checkpoint to an offline ONNX bundle."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the fine-tuned Hugging Face checkpoint.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help="Directory used to assemble the ONNX deployment bundle.",
    )
    parser.add_argument(
        "--sample-text",
        default="我想查一下本月信用卡还款金额",
        help="Sample text used to trace the ONNX graph.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override the export max_length. Defaults to train_config.json or 64.",
    )
    parser.add_argument(
        "--onnx-file",
        default="model.onnx",
        help="Filename for the exported ONNX model.",
    )
    parser.add_argument(
        "--quantized-onnx-file",
        default="model.int8.onnx",
        help="Filename for the exported INT8 ONNX model.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 dynamic quantization.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip creating a tar.gz archive for transfer into the internal network.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_max_length(model_dir: Path, override: int | None) -> int:
    if override is not None:
        return override
    config_path = model_dir / "train_config.json"
    if config_path.exists():
        train_config = load_json(config_path)
        if "max_length" in train_config:
            return int(train_config["max_length"])
    return 64


def build_dummy_inputs(tokenizer, sample_text: str, max_length: int):
    encoded = tokenizer(
        sample_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_names = [
        name
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in encoded
    ]
    example_inputs = tuple(encoded[name] for name in input_names)
    dynamic_axes = {name: {0: "batch_size", 1: "seq_len"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch_size"}
    return encoded, input_names, example_inputs, dynamic_axes


def copy_optional_file(source: Path, target: Path) -> None:
    if source.exists():
        target.write_bytes(source.read_bytes())


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "label_mapping.json").exists():
        raise FileNotFoundError(
            f"label_mapping.json is missing in model directory: {model_dir}"
        )

    max_length = resolve_max_length(model_dir, args.max_length)
    label_mapping = load_json(model_dir / "label_mapping.json")
    train_config = load_json(model_dir / "train_config.json") if (model_dir / "train_config.json").exists() else {}

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).cpu().eval()
    export_model = SequenceClassifierExportWrapper(model)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(bundle_dir)
    save_json(bundle_dir / "label_mapping.json", label_mapping)
    if train_config:
        save_json(bundle_dir / "train_config.json", train_config)
    copy_optional_file(model_dir / "best_valid_metrics.json", bundle_dir / "best_valid_metrics.json")

    _, input_names, example_inputs, dynamic_axes = build_dummy_inputs(
        tokenizer,
        args.sample_text,
        max_length,
    )

    onnx_path = bundle_dir / args.onnx_file
    quantized_path = bundle_dir / args.quantized_onnx_file

    torch.onnx.export(
        export_model,
        example_inputs,
        onnx_path.as_posix(),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    onnx_model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(onnx_model)

    quantized_file: str | None = None
    if not args.no_quantize:
        quantize_dynamic(
            model_input=onnx_path.as_posix(),
            model_output=quantized_path.as_posix(),
            weight_type=QuantType.QInt8,
        )
        quantized_file = quantized_path.name

    manifest = {
        "task": "intent-classification",
        "source_model": train_config.get("model_name_or_path", "hfl/minirbt-h256"),
        "exported_from": model_dir.as_posix(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "max_length": max_length,
        "input_names": input_names,
        "onnx_file": onnx_path.name,
        "quantized_onnx_file": quantized_file,
        "label2id": label_mapping["label2id"],
        "id2label": label_mapping["id2label"],
        "normalization": {
            "unicode_normalize": "NFKC",
            "remove_whitespace": True,
        },
        "external_network_training": True,
        "internal_network_ready": True,
    }
    save_json(bundle_dir / "manifest.json", manifest)

    archive_path = bundle_dir.with_suffix(".tar.gz")
    if not args.no_archive:
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(bundle_dir, arcname=bundle_dir.name)

    summary = {
        "bundle_dir": bundle_dir.as_posix(),
        "onnx_file": onnx_path.as_posix(),
        "quantized_onnx_file": quantized_path.as_posix() if quantized_file else None,
        "archive": archive_path.as_posix() if not args.no_archive else None,
        "input_names": input_names,
        "max_length": max_length,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()