# MiniRBT Intent Classification Demo

This directory contains a minimal end-to-end pipeline for the intent classification dataset in `data/意图识别/train.jsonl`.

The pipeline is split into two environments:

1. External network training environment
2. Internal network ONNX inference environment

## Files

- `train_minirbt_intent.py`: fine-tune `hfl/minirbt-h256` on the intent dataset and save the best Hugging Face checkpoint locally.
- `export_onnx_bundle.py`: export the local PyTorch checkpoint to ONNX, optionally quantize to INT8, and assemble an offline bundle.
- `infer_onnx_offline.py`: run offline inference from the ONNX bundle without any external network dependency.
- `requirements-train.txt`: minimal packages for training and ONNX export.
- `requirements-infer.txt`: minimal packages for internal ONNX inference.

## External Network Training

Create an isolated environment first.

```bash
conda create -n minirbt_intent python=3.10 -y
conda activate minirbt_intent
pip install -r test_codes/requirements-train.txt
```

Run training.

```bash
python test_codes/train_minirbt_intent.py \
  --dataset-path data/意图识别/train.jsonl \
  --output-dir artifacts/minirbt_intent_training
```

After training, export the best checkpoint to an intranet-ready ONNX bundle.

```bash
python test_codes/export_onnx_bundle.py \
  --model-dir artifacts/minirbt_intent_training/best_model \
  --bundle-dir artifacts/minirbt_intent_onnx_bundle
```

The export step writes:

- `model.onnx`
- `model.int8.onnx`
- tokenizer files
- `label_mapping.json`
- `manifest.json`
- `bundle.tar.gz`

You can copy either the directory or the `tar.gz` archive into the internal network.

## Internal Network Deployment

On the internal network machine, unpack the bundle if needed and create a minimal inference environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r test_codes/requirements-infer.txt
```

Run offline inference.

```bash
python test_codes/infer_onnx_offline.py \
  --bundle-dir artifacts/minirbt_intent_onnx_bundle \
  --text "我这个月信用卡还要还多少钱？"
```

Batch inference from a JSONL file is also supported.

```bash
python test_codes/infer_onnx_offline.py \
  --bundle-dir artifacts/minirbt_intent_onnx_bundle \
  --input-jsonl data/意图识别/train.jsonl
```

## Notes

- Training can download `hfl/minirbt-h256` from Hugging Face when the external network is available.
- ONNX inference loads everything from local files only.
- The inference script does not depend on PyTorch.
- For internal deployment, the recommended artifact is the quantized `model.int8.onnx` plus tokenizer and label mapping files.