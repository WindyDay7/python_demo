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
        # Hugging Face SequenceClassification 模型默认返回的是一个包含 loss、logits 等字段
        # 的 ModelOutput 对象，而 torch.onnx.export 更适合导出“张量到张量”的稳定接口。
        # 这个 wrapper 的作用就是把复杂对象压平成纯 logits 输出，确保 ONNX 图的输入输出
        # 语义简单明确，方便后续 onnxruntime 侧直接调用。
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
    # ONNX 导出本质上是“沿着一次真实前向计算轨迹追踪计算图”。因此这里必须构造一组
    # 与真实推理一致的样例输入，至少包括 input_ids / attention_mask，若底层模型需要，
    # 也可能包含 token_type_ids。padding 到固定长度是为了让导出过程覆盖完整序列维度。
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
    # dynamic_axes 告诉 ONNX：batch 维和序列长度维不是常量。这样导出的模型就不会被
    # 锁死在某一个固定 batch 或固定 seq_len 上，离线推理时可以处理不同长度的输入。
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

    # 导出阶段必须同时加载 tokenizer 和 fine-tuned 后的分类模型权重。
    # tokenizer 决定文本如何切分成 token id，模型权重则决定这些 token 最终如何映射成
    # 分类 logits。两者必须来自同一训练产物目录，才能保证 bundle 内部自洽。
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).cpu().eval()
    # export_model 只暴露 logits，避免把训练阶段才需要的 loss 等输出也带进部署图中。
    export_model = SequenceClassifierExportWrapper(model)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    # bundle 不仅保存 ONNX 文件，还要保存 tokenizer 配置和标签映射，这样目标环境在
    # 没有外网、没有训练代码的情况下，仍然可以完整重建“文本 -> 张量 -> 标签”的闭环。
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

    # torch.onnx.export 会运行一次前向，并把其中涉及的算子导出成 ONNX 计算图。
    # output_names 固定为 logits，便于推理脚本按稳定名字取结果；opset_version 需要与
    # 当前 transformers / onnxruntime 支持的算子集合兼容。
    torch.onnx.export(
        export_model,
        example_inputs,
        onnx_path.as_posix(),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    # 导出后立即做 ONNX 结构检查，可以尽早发现不兼容算子、图结构损坏或 shape 元数据异常。
    onnx_model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(onnx_model)

    quantized_file: str | None = None
    if not args.no_quantize:
        # 动态量化主要压缩线性层权重，把部分 FP32 权重转成 INT8。对 CPU 推理场景，
        # 这通常能显著降低模型体积，并在很多环境下获得更好的吞吐或更低延迟。
        # 这里保留原始 ONNX，再额外生成量化版，便于后续对效果和性能做对比回退。
        quantize_dynamic(
            model_input=onnx_path.as_posix(),
            model_output=quantized_path.as_posix(),
            weight_type=QuantType.QInt8,
        )
        quantized_file = quantized_path.name

    # manifest 是部署 bundle 的“说明书”。推理脚本依赖它找到模型文件、最大长度、输入名，
    # 并明确训练阶段采用了什么文本归一化和标签映射，从而避免部署端猜测这些约定。
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
        # 对内网交付场景，把整个 bundle 打成 tar.gz 更方便搬运；因为 tokenizer 文件、
        # manifest、标签映射和模型文件必须一起传输，缺一项都会导致离线推理不可复现。
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