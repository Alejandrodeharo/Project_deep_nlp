from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_functions import build_model, test_step
from utils import (
    NERDataset,
    SentimentDataset,
    load_checkpoint,
    load_json,
    ner_collate_fn,
    normalize_sentiment_examples,
    sentiment_collate_fn,
)


class NERSpanAwareLoss(nn.Module):
    """
    Loss para NER que penaliza más los falsos negativos de entidad.

    Combina:
    1. CrossEntropy ponderada por clase.
    2. Penalización extra cuando el gold es una entidad (B-* o I-*)
       y el modelo asigna mucha probabilidad a la clase O.

    Se usa automáticamente solo si el checkpoint de NER fue entrenado con ella.
    """

    def __init__(
        self,
        tag2idx: Dict[str, int],
        class_weights: torch.Tensor | None = None,
        lambda_miss: float = 1.5,
    ) -> None:
        super().__init__()
        self.tag2idx = tag2idx
        self.pad_idx = tag2idx["<PAD>"]
        self.o_idx = tag2idx["O"]
        self.lambda_miss = float(lambda_miss)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, T, C]
        labels: [B, T]
        """
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
        entity_mask = valid_mask & (labels != self.o_idx)

        if entity_mask.any():
            miss_penalty = -torch.log((1.0 - p_o[entity_mask]).clamp(min=1e-8)).mean()
        else:
            miss_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)

        return ce_loss + self.lambda_miss * miss_penalty


def build_ner_class_weights(
    tag2idx: Dict[str, int],
    weight_o: float = 1.0,
    weight_b: float = 2.0,
    weight_i: float = 3.0,
) -> torch.Tensor:
    """
    Construye pesos por clase para NER.
    """
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--metrics_output_path",
        type=str,
        default=None,
        help="Ruta opcional para guardar las métricas de test en JSON.",
    )
    return parser.parse_args()


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

        # Sentiment usa siempre su propia loss, independiente de NER.
        criterion = nn.CrossEntropyLoss()
        metric_name = "accuracy"
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

        # NER usa automáticamente la loss correcta según cómo fue entrenado el checkpoint.
        if checkpoint_config.get("use_span_aware_ner_loss", False):
            class_weights = build_ner_class_weights(
                metadata["tag2idx"],
                weight_o=checkpoint_config.get("ner_weight_o", 1.0),
                weight_b=checkpoint_config.get("ner_weight_b", 2.0),
                weight_i=checkpoint_config.get("ner_weight_i", 3.0),
            )

            criterion = NERSpanAwareLoss(
                tag2idx=metadata["tag2idx"],
                class_weights=class_weights,
                lambda_miss=checkpoint_config.get("ner_lambda_miss", 1.5),
            )
        else:
            # Si el checkpoint NER no fue entrenado con la loss nueva,
            # se mantiene coherencia usando CE estándar.
            criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_idx)

        metric_name = "token_accuracy"

    else:
        raise ValueError(f"Unsupported task: {task}")

    return dataloader, criterion, metric_name, pad_tag_idx


def save_metrics(path: str, data: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    dataloader, criterion, metric_name, pad_tag_idx = build_eval_dataloader(
        task=task,
        data_path=args.data_path,
        metadata=metadata,
        checkpoint_config=cfg,
        batch_size=args.batch_size,
    )
    criterion = criterion.to(device)

    loss, metric = test_step(
        model=model,
        test_data=dataloader,
        criterion=criterion,
        device=device,
        task=task,
        pad_tag_idx=pad_tag_idx,
    )

    print(f"Task: {task}")
    print(f"Model: {model_name}")
    print(f"Loss: {loss:.4f}")
    print(f"{metric_name}: {metric:.4f}")

    if args.metrics_output_path is not None:
        results = {
            "task": task,
            "model_name": model_name,
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "loss": loss,
            metric_name: metric,
        }
        save_metrics(args.metrics_output_path, results)
        print(f"Saved test metrics to: {args.metrics_output_path}")


if __name__ == "__main__":
    main()