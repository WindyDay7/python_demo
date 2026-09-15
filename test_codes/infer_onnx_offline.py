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
    # 这里的文本归一化需要与训练/导出阶段保持一致，否则同一句用户输入可能在
    # 训练时被编码成一种 token 序列，在推理时又被编码成另一种 token 序列，
    # 最终造成线上效果与离线验证不一致。
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def softmax(logits: np.ndarray) -> np.ndarray:
    # ONNX 分类模型输出的 logits 是“未归一化分数”，它们只表达各类别相对强弱，
    # 还不是概率。这里先减去每行最大值做数值稳定化，再做 softmax，把输出转成
    # 每个类别概率之和为 1 的分布，便于返回 top-k 结果和置信度分数。
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

    # manifest 里优先记录量化后的 INT8 模型，因为它通常体积更小、CPU 推理更快，
    # 更适合内网离线部署。如果 bundle 没有量化版本，再回退到原始 FP32 ONNX 模型。
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
            # tokenizer 输出里常见的键包括 input_ids、attention_mask，某些 BERT 系模型
            # 还会带 token_type_ids。这里不是把 tokenizer 的所有输出一股脑送进 ONNX，
            # 而是以 ONNX 图里声明的输入为准逐个对齐，避免输入名或输入数量不匹配。
            inputs[name] = encoded[name]
            continue
        if name == "token_type_ids":
            # MiniRBT/部分中文 BERT 在单句分类场景里通常并不真正区分句对 segment，
            # 但导出的 ONNX 图可能仍然保留 token_type_ids 这个输入。若 tokenizer 未返回，
            # 用全 0 张量补齐，语义上表示“所有 token 都属于同一段文本”。
            inputs[name] = np.zeros_like(encoded["input_ids"], dtype=np.int64)
            continue
        raise KeyError(f"Tokenizer output is missing required ONNX input: {name}")
    return inputs


def top_k_predictions(
    probabilities: np.ndarray,
    id2label: dict[int, str],
    top_k: int,
) -> list[dict[str, Any]]:
    # 概率向量长度等于类别数。这里先排序取概率最高的 k 个类别，再通过训练时
    # 固定下来的 id2label 做反查，保证输出标签名称与训练时的类别编号严格一致。
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
    # label_mapping.json 保留的是训练阶段生成的稳定标签编号映射。推理时必须复用这份
    # 映射，而不能重新按字符串排序生成，否则 logits 第 i 维会被解释成错误的业务标签。
    id2label = {int(key): value for key, value in mapping["id2label"].items()}

    model_path = resolve_model_path(bundle_dir, manifest, args.model_file)
    max_length = args.max_length or int(manifest.get("max_length", 64))
    texts = read_texts(args)

    # tokenizer 从 bundle_dir 本地加载，意味着这里完全不依赖外网 Hugging Face Hub。
    # 更重要的是，分词器词表、special tokens、normalizer 配置都和训练产物保持一致，
    # 这是离线部署可复现的关键前提。
    tokenizer = AutoTokenizer.from_pretrained(bundle_dir, local_files_only=True)
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="np",
    )

    # InferenceSession 会读取 ONNX 图，并根据 provider 选择实际执行后端。
    # 在当前脚本默认配置下，一般使用 CPUExecutionProvider，因此部署环境只需要
    # onnxruntime CPU 版本即可完成前向计算。
    session = ort.InferenceSession(
        model_path.as_posix(),
        providers=[args.provider],
    )
    session_inputs = build_session_inputs(session, encoded)

    # 我们导出 ONNX 时把输出名固定为 logits，所以这里直接按名字取回分类头输出。
    # logits 的形状通常是 [batch_size, num_labels]。
    logits = session.run(["logits"], session_inputs)[0]
    probabilities = softmax(logits)

    for text, sample_probabilities in zip(texts, probabilities):
        # 每条文本独立做 top-k 解码，便于既返回最终预测标签，也保留候选类别及分数，
        # 后续如果要接人工复核或阈值策略，可以直接使用这里的完整结果。
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