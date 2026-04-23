from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_functions import build_model
from utils import (
    NERDataset,
    SentimentDataset,
    load_checkpoint,
    load_json,
    ner_collate_fn,
    normalize_sentiment_examples,
    sentiment_collate_fn,
)


def build_ner_class_weights(
    tag2idx: dict[str, int],
    weight_o: float = 1.0,
    weight_b: float = 2.0,
    weight_i: float = 2.5,
) -> torch.Tensor:
    weights = torch.ones(len(tag2idx), dtype=torch.float32)

    for tag, idx in tag2idx.items():
        if tag == "<PAD>":
            weights[idx] = 0.0
        elif tag == "O":
            weights[idx] = weight_o
        elif tag.startswith("B-"):
            weights[idx] = weight_b
        elif tag.startswith("I-"):
            weights[idx] = weight_i
        else:
            weights[idx] = 1.0

    return weights


def build_valid_transition_mask(tag2idx: dict[str, int]) -> torch.Tensor:
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    n = len(tag2idx)
    valid = torch.zeros((n, n), dtype=torch.float32)

    def split_tag(tag: str):
        if tag in {"O", "<PAD>"}:
            return tag, None
        if "-" not in tag:
            return tag, None
        prefix, label = tag.split("-", 1)
        return prefix, label

    for prev_idx in range(n):
        prev_tag = idx2tag[prev_idx]
        prev_prefix, prev_label = split_tag(prev_tag)

        for curr_idx in range(n):
            curr_tag = idx2tag[curr_idx]
            curr_prefix, curr_label = split_tag(curr_tag)

            is_valid = False

            if curr_tag == "<PAD>":
                is_valid = True
            elif curr_tag == "O":
                is_valid = True
            elif curr_prefix == "B":
                is_valid = True
            elif curr_prefix == "I":
                if prev_prefix in {"B", "I"} and prev_label == curr_label:
                    is_valid = True
                else:
                    is_valid = False
            else:
                is_valid = False

            valid[prev_idx, curr_idx] = 1.0 if is_valid else 0.0

    return valid


class NERStructureAwareLoss(nn.Module):
    def __init__(
        self,
        tag2idx: dict[str, int],
        class_weights: torch.Tensor | None = None,
        lambda_miss: float = 1.2,
        lambda_fp: float = 1.0,
        lambda_transition: float = 0.8,
    ) -> None:
        super().__init__()
        self.tag2idx = tag2idx
        self.pad_idx = tag2idx["<PAD>"]
        self.o_idx = tag2idx["O"]
        self.lambda_miss = float(lambda_miss)
        self.lambda_fp = float(lambda_fp)
        self.lambda_transition = float(lambda_transition)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        valid_transition_mask = build_valid_transition_mask(tag2idx)
        self.register_buffer("valid_transition_mask", valid_transition_mask)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            ignore_index=self.pad_idx,
        )

        probs = torch.softmax(logits, dim=-1)
        p_o = probs[..., self.o_idx]

        valid_mask = labels != self.pad_idx
        gold_entity_mask = valid_mask & (labels != self.o_idx)
        gold_o_mask = valid_mask & (labels == self.o_idx)

        if gold_entity_mask.any():
            miss_penalty = -torch.log((1.0 - p_o[gold_entity_mask]).clamp(min=1e-8)).mean()
        else:
            miss_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)

        p_entity = 1.0 - p_o
        if gold_o_mask.any():
            fp_penalty = -torch.log((1.0 - p_entity[gold_o_mask]).clamp(min=1e-8)).mean()
        else:
            fp_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if logits.size(1) > 1:
            prev_probs = probs[:, :-1, :]
            curr_probs = probs[:, 1:, :]

            pair_probs = prev_probs.unsqueeze(-1) * curr_probs.unsqueeze(-2)
            invalid_mask = 1.0 - self.valid_transition_mask

            invalid_mass = (pair_probs * invalid_mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))

            transition_valid_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            if transition_valid_mask.any():
                transition_penalty = invalid_mass[transition_valid_mask].mean()
            else:
                transition_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            transition_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)

        return (
            ce_loss
            + self.lambda_miss * miss_penalty
            + self.lambda_fp * fp_penalty
            + self.lambda_transition * transition_penalty
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model and export detailed predictions")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/test_predictions.json",
        help="Ruta del JSON con métricas y predicciones por sample.",
    )
    return parser.parse_args()


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def decode_bio_predictions_from_tokens(tokens: List[str], tags: List[str]) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    current_label = None
    current_tokens: List[str] = []

    def flush() -> None:
        nonlocal current_label, current_tokens
        if current_label is not None and current_tokens:
            entities.append({"text": " ".join(current_tokens), "label": current_label})
        current_label = None
        current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag in {"O", "<PAD>"}:
            flush()
            continue

        if "-" not in tag:
            flush()
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B":
            flush()
            current_label = label
            current_tokens = [token]
        elif prefix == "I" and current_label == label:
            current_tokens.append(token)
        else:
            flush()
            if prefix == "I":
                current_label = label
                current_tokens = [token]

    flush()
    return entities


def compute_entity_metrics_from_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    total_pred = 0
    total_gold = 0
    total_correct = 0
    total_exact = 0

    for item in predictions:
        gold_entities = {(ent["text"], ent["label"]) for ent in item["gold_entities"]}
        pred_entities = {(ent["text"], ent["label"]) for ent in item["predicted_entities"]}

        total_gold += len(gold_entities)
        total_pred += len(pred_entities)
        total_correct += len(gold_entities & pred_entities)
        total_exact += int(gold_entities == pred_entities)

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = total_exact / len(predictions) if predictions else 0.0

    return {
        "entity_precision": float(precision),
        "entity_recall": float(recall),
        "entity_f1": float(f1),
        "sample_exact_match": float(exact_match),
    }


def build_eval_dataloader(
    task: str,
    data_path: str,
    metadata: dict,
    checkpoint_config: dict,
    batch_size: int,
):
    examples = load_json(data_path)

    if task == "sentiment":
        examples = normalize_sentiment_examples(examples)

        dataset = SentimentDataset(
            examples=examples,
            word2idx=metadata["word2idx"],
            label2idx=metadata["label2idx"],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=sentiment_collate_fn,
        )

        criterion = nn.CrossEntropyLoss()
        pad_tag_idx = 0

    elif task == "ner":
        dataset = NERDataset(
            examples=examples,
            word2idx=metadata["word2idx"],
            tag2idx=metadata["tag2idx"],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ner_collate_fn,
        )

        pad_tag_idx = metadata["tag2idx"]["<PAD>"]

        class_weights = build_ner_class_weights(
            metadata["tag2idx"],
            weight_o=checkpoint_config.get("ner_weight_o", 1.0),
            weight_b=checkpoint_config.get("ner_weight_b", 2.0),
            weight_i=checkpoint_config.get("ner_weight_i", 2.5),
        )

        criterion = NERStructureAwareLoss(
            tag2idx=metadata["tag2idx"],
            class_weights=class_weights,
            lambda_miss=checkpoint_config.get("ner_lambda_miss", 1.2),
            lambda_fp=checkpoint_config.get("ner_lambda_fp", 1.0),
            lambda_transition=checkpoint_config.get("ner_lambda_transition", 0.8),
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    return examples, dataloader, criterion, pad_tag_idx


@torch.no_grad()
def evaluate_sentiment(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    idx2label: Dict[int, Any],
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_batches = 0
    predictions: List[Dict[str, Any]] = []

    for batch in dataloader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        lengths = batch[2].to(device)
        texts = batch[3]

        logits = model(inputs, lengths)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)

        total_loss += loss.item()
        total_correct += (pred_ids == labels).sum().item()
        total_items += labels.numel()
        total_batches += 1

        for i in range(inputs.size(0)):
            pred_idx = int(pred_ids[i].item())
            gold_idx = int(labels[i].item())

            predictions.append(
                {
                    "text": texts[i],
                    "gold_label": idx2label[gold_idx],
                    "predicted_label": idx2label[pred_idx],
                    "correct": pred_idx == gold_idx,
                    "probabilities": {
                        str(idx2label[j]): float(probs[i, j].item())
                        for j in range(probs.size(1))
                    },
                }
            )

    mean_loss = total_loss / max(total_batches, 1)
    accuracy = total_correct / max(total_items, 1)

    metrics = {
        "accuracy": float(accuracy),
    }
    return float(mean_loss), metrics, predictions


@torch.no_grad()
def evaluate_ner(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    idx2tag: Dict[int, str],
    pad_tag_idx: int,
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_batches = 0
    predictions: List[Dict[str, Any]] = []

    for batch in dataloader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        lengths = batch[2].to(device)
        token_lists = batch[3]
        text_list = batch[4]

        logits = model(inputs, lengths)
        loss = criterion(logits, labels)

        pred_ids = logits.argmax(dim=-1)

        mask = labels != pad_tag_idx
        total_loss += loss.item()
        total_correct += ((pred_ids == labels) & mask).sum().item()
        total_items += mask.sum().item()
        total_batches += 1

        for i in range(inputs.size(0)):
            seq_len = int(lengths[i].item())
            tokens = token_lists[i]

            gold_tag_ids = labels[i][:seq_len].tolist()
            pred_tag_ids = pred_ids[i][:seq_len].tolist()

            gold_tags = [idx2tag[int(tag_id)] for tag_id in gold_tag_ids]
            pred_tags = [idx2tag[int(tag_id)] for tag_id in pred_tag_ids]

            gold_entities = decode_bio_predictions_from_tokens(tokens, gold_tags)
            pred_entities = decode_bio_predictions_from_tokens(tokens, pred_tags)

            token_correct = sum(int(g == p) for g, p in zip(gold_tags, pred_tags))
            token_total = len(gold_tags)
            token_acc = token_correct / max(token_total, 1)

            predictions.append(
                {
                    "text": text_list[i],
                    "tokens": tokens,
                    "gold_tags": gold_tags,
                    "predicted_tags": pred_tags,
                    "gold_entities": gold_entities,
                    "predicted_entities": pred_entities,
                    "token_accuracy_sample": token_acc,
                }
            )

    mean_loss = total_loss / max(total_batches, 1)
    token_accuracy = total_correct / max(total_items, 1)

    entity_metrics = compute_entity_metrics_from_predictions(predictions)
    metrics = {
        "token_accuracy": float(token_accuracy),
        **entity_metrics,
    }

    return float(mean_loss), metrics, predictions


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)

    task = ckpt["task"]
    model_name = ckpt["model_name"]
    cfg = ckpt["config"]
    metadata = ckpt["metadata"]

    model, _ = build_model(
        task=task,
        model_name=model_name,
        vocab_size=len(metadata["word2idx"]),
        num_outputs=metadata["num_outputs"],
        embed_dim=cfg.get("embed_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    _, dataloader, criterion, pad_tag_idx = build_eval_dataloader(
        task=task,
        data_path=args.data_path,
        metadata=metadata,
        checkpoint_config=cfg,
        batch_size=args.batch_size,
    )
    criterion = criterion.to(device)

    if task == "sentiment":
        loss, metrics, predictions = evaluate_sentiment(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            idx2label=metadata["idx2label"],
        )
    elif task == "ner":
        loss, metrics, predictions = evaluate_ner(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            idx2tag=metadata["idx2tag"],
            pad_tag_idx=pad_tag_idx,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    results = {
        "task": task,
        "model_name": model_name,
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "metrics": {
            "loss": loss,
            **metrics,
        },
        "predictions": predictions,
    }

    save_json(args.output_path, results)

    print(f"Task: {task}")
    print(f"Model: {model_name}")
    print(f"Loss: {loss:.4f}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"Saved detailed predictions to: {args.output_path}")


if __name__ == "__main__":
    main()