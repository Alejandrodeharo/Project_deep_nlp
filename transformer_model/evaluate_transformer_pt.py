from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from transformer_inference import decode_bio_predictions
from utils import tokenize_with_offsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="test_predictions.json")
    return parser.parse_args()


def load_sentiment_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    metadata = ckpt["metadata"]
    model_name = ckpt["model_name"]

    id2value = {int(k): int(v) for k, v in metadata["id2value"].items()}
    id2label_name = {int(k): str(v) for k, v in metadata["id2label_name"].items()}

    config = AutoConfig.from_pretrained(model_name, num_labels=len(id2value))
    config.id2label = {idx: id2label_name[idx] for idx in sorted(id2label_name)}
    config.label2id = {v: k for k, v in config.id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer, metadata, id2value, id2label_name


def load_ner_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    metadata = ckpt["metadata"]
    model_name = ckpt["model_name"]

    id2tag = {int(k): str(v) for k, v in metadata["id2tag"].items()}
    tag2id = {v: k for k, v in id2tag.items()}

    config = AutoConfig.from_pretrained(model_name, num_labels=len(id2tag))
    config.id2label = {idx: id2tag[idx] for idx in sorted(id2tag)}
    config.label2id = tag2id

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer, metadata, id2tag


def normalize_entity(entity: dict):
    return (entity["text"].strip(), entity["label"].strip())


def evaluate_sentiment(model, tokenizer, metadata, id2value, examples, device):
    max_length = int(metadata.get("max_length", 256))

    results = []
    correct = 0
    total = 0

    for item in examples:
        encoded = tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = int(probs.argmax().item())
        pred_value = id2value[pred_idx]

        row = {
            "match_id": item.get("match_id"),
            "prediction": pred_value,
            "confidence": float(probs[pred_idx].item()),
        }

        if "sentiment" in item:
            row["gold"] = int(item["sentiment"])
            total += 1
            if pred_value == int(item["sentiment"]):
                correct += 1

        results.append(row)

    metrics = {}
    if total > 0:
        metrics["accuracy"] = correct / total

    return results, metrics


def evaluate_ner(model, tokenizer, metadata, id2tag, examples, device):
    max_length = int(metadata.get("max_length", 256))

    results = []
    tp = 0
    fp = 0
    fn = 0

    for item in examples:
        text = item["text"]
        token_spans = tokenize_with_offsets(text)
        tokens = [token for token, _, _ in token_spans]

        if not tokens:
            pred_entities = []
        else:
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = model(**encoded).logits[0]

            pred_ids = logits.argmax(dim=-1).tolist()

            tokenizer_output = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
            )
            word_id_sequence = tokenizer_output.word_ids()

            word_predictions = {}
            for token_idx, word_id in enumerate(word_id_sequence):
                if word_id is None:
                    continue
                if word_id not in word_predictions:
                    word_predictions[word_id] = pred_ids[token_idx]

            tags = []
            for word_index in range(len(tokens)):
                pred_id = word_predictions.get(word_index, 1)
                tags.append(id2tag.get(int(pred_id), "O"))

            pred_entities = decode_bio_predictions(text=text, token_spans=token_spans, tags=tags)

        row = {
            "match_id": item.get("match_id"),
            "prediction_entities": pred_entities,
        }

        if "entities" in item:
            gold_set = {normalize_entity(x) for x in item["entities"]}
            pred_set = {normalize_entity(x) for x in pred_entities}

            item_tp = len(gold_set & pred_set)
            item_fp = len(pred_set - gold_set)
            item_fn = len(gold_set - pred_set)

            tp += item_tp
            fp += item_fp
            fn += item_fn

            row["gold_entities"] = item["entities"]

        results.append(row)

    metrics = {}
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["entity_precision"] = precision
    metrics["entity_recall"] = recall
    metrics["entity_f1"] = f1

    return results, metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    ckpt = torch.load(args.checkpoint, map_location=device)
    task = ckpt["task"]

    if task == "sentiment":
        model, tokenizer, metadata, id2value, id2label_name = load_sentiment_model(args.checkpoint, device)
        results, metrics = evaluate_sentiment(model, tokenizer, metadata, id2value, examples, device)
    elif task == "ner":
        model, tokenizer, metadata, id2tag = load_ner_model(args.checkpoint, device)
        results, metrics = evaluate_ner(model, tokenizer, metadata, id2tag, examples, device)
    else:
        raise ValueError(f"Unsupported task: {task}")

    output = {
        "task": task,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "predictions": results,
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()