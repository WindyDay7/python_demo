from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "意图识别" / "train.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "minirbt_intent_training"
DEFAULT_MODEL_NAME = "hfl/minirbt-h256"


@dataclass
class TrainConfig:
    dataset_path: Path
    output_dir: Path
    model_name_or_path: str
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    warmup_ratio: float
    grad_accum_steps: int
    max_grad_norm: float
    valid_ratio: float
    test_ratio: float
    seed: int
    local_files_only: bool


class IntentDataset(Dataset):
    def __init__(self, texts: list[str], label_ids: list[int] | None = None) -> None:
        self.texts = texts
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {"text": self.texts[index]}
        if self.label_ids is not None:
            item["label"] = self.label_ids[index]
        return item


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Fine-tune MiniRBT-H256 on the intent classification dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the input JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save checkpoints and metrics.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load local model/tokenizer files and do not access the network.",
    )
    args = parser.parse_args()

    if args.valid_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("valid_ratio and test_ratio must both be greater than 0.")
    if args.valid_ratio + args.test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be less than 1.")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be greater than 0.")

    return TrainConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        local_files_only=args.local_files_only,
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_intent_dataframe(dataset_path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "input" not in row or "output" not in row:
                raise KeyError(
                    f"Line {line_number} is missing required keys 'input'/'output'."
                )
            records.append(
                {
                    "text": normalize_text(str(row["input"])),
                    "label": str(row["output"]),
                }
            )

    if not records:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    frame = pd.DataFrame(records)
    frame["text_len"] = frame["text"].str.len()
    return frame


def split_dataframe(
    frame: pd.DataFrame,
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout_ratio = valid_ratio + test_ratio
    train_df, holdout_df = train_test_split(
        frame,
        test_size=holdout_ratio,
        stratify=frame["label"],
        random_state=seed,
    )

    test_ratio_in_holdout = test_ratio / holdout_ratio
    valid_df, test_df = train_test_split(
        holdout_df,
        test_size=test_ratio_in_holdout,
        stratify=holdout_df["label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = [int(item["label"]) for item in batch if "label" in item]
        if labels:
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def evaluate(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
    id2label: dict[int, str],
) -> dict[str, Any]:
    ordered_label_ids = list(range(len(id2label)))
    target_names = [id2label[index] for index in ordered_label_ids]

    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().cpu()))
            logits = outputs.logits.detach().cpu()
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
            references.extend(batch["labels"].detach().cpu().tolist())

    report_dict = classification_report(
        references,
        predictions,
        labels=ordered_label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        references,
        predictions,
        labels=ordered_label_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    return {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_f1": float(f1_score(references, predictions, average="macro")),
        "report": report_dict,
        "report_text": report_text,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    return {
        "dataset_path": config.dataset_path.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "model_name_or_path": config.model_name_or_path,
        "max_length": config.max_length,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "grad_accum_steps": config.grad_accum_steps,
        "max_grad_norm": config.max_grad_norm,
        "valid_ratio": config.valid_ratio,
        "test_ratio": config.test_ratio,
        "seed": config.seed,
        "local_files_only": config.local_files_only,
    }


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = choose_device()

    frame = load_intent_dataframe(config.dataset_path)
    train_df, valid_df, test_df = split_dataframe(
        frame,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    labels = sorted(frame["label"].unique())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        local_files_only=config.local_files_only,
    )

    train_dataset = IntentDataset(
        texts=train_df["text"].tolist(),
        label_ids=[label2id[label] for label in train_df["label"].tolist()],
    )
    valid_dataset = IntentDataset(
        texts=valid_df["text"].tolist(),
        label_ids=[label2id[label] for label in valid_df["label"].tolist()],
    )
    test_dataset = IntentDataset(
        texts=test_df["text"].tolist(),
        label_ids=[label2id[label] for label in test_df["label"].tolist()],
    )

    collate_fn = build_collate_fn(tokenizer, config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        local_files_only=config.local_files_only,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
    total_training_steps = update_steps_per_epoch * config.num_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    output_dir = config.output_dir
    best_model_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summary = {
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "label_distribution": {
            "train": train_df["label"].value_counts().to_dict(),
            "valid": valid_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
    }
    save_json(output_dir / "split_summary.json", split_summary)

    best_macro_f1 = -1.0
    history: list[dict[str, Any]] = []

    for epoch_index in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss_sum = 0.0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch_index + 1}/{config.num_epochs}",
            leave=False,
        )

        for step_index, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Training loss is missing from the model output.")

            raw_loss = float(loss.detach().cpu())
            epoch_loss_sum += raw_loss

            (loss / config.grad_accum_steps).backward()

            should_step = (
                step_index % config.grad_accum_steps == 0
                or step_index == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress.set_postfix(loss=f"{raw_loss:.4f}")

        train_loss = epoch_loss_sum / max(len(train_loader), 1)
        valid_metrics = evaluate(model, valid_loader, device, id2label)

        epoch_record = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
        }
        history.append(epoch_record)

        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            save_json(
                best_model_dir / "label_mapping.json",
                {
                    "label2id": label2id,
                    "id2label": {str(key): value for key, value in id2label.items()},
                },
            )
            save_json(best_model_dir / "train_config.json", config_to_dict(config))
            save_json(
                best_model_dir / "best_valid_metrics.json",
                {
                    "epoch": epoch_index + 1,
                    "loss": valid_metrics["loss"],
                    "accuracy": valid_metrics["accuracy"],
                    "macro_f1": valid_metrics["macro_f1"],
                    "report": valid_metrics["report"],
                },
            )
            (best_model_dir / "best_valid_report.txt").write_text(
                valid_metrics["report_text"],
                encoding="utf-8",
            )

        print(
            json.dumps(
                epoch_record,
                ensure_ascii=False,
            )
        )

    save_json(output_dir / "training_history.json", {"history": history})

    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_model_dir,
        local_files_only=True,
    ).to(device)
    test_metrics = evaluate(best_model, test_loader, device, id2label)

    save_json(
        output_dir / "test_metrics.json",
        {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "report": test_metrics["report"],
        },
    )
    (output_dir / "test_report.txt").write_text(
        test_metrics["report_text"],
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "device": str(device),
                "best_model_dir": best_model_dir.as_posix(),
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()