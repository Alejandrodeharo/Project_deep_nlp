from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import MODEL_REGISTRY
from utils import (
    NERDataset,
    SentimentDataset,
    build_label_mappings,
    build_tag_mappings,
    build_vocab_from_ner_examples,
    build_vocab_from_texts,
    load_json,
    ner_collate_fn,
    normalize_sentiment_examples,
    sentiment_collate_fn,
    train_test_split_manual,
)


def build_dataloaders(
    task: str,
    data_path: str,
    test_size: float,
    seed: int,
    batch_size: int,
    max_vocab_size: int,
) -> Tuple[DataLoader, DataLoader, Dict]:
    data = load_json(data_path)

    if task == "sentiment":
        data = normalize_sentiment_examples(data)

    train_examples, val_examples = train_test_split_manual(
        data=data,
        test_size=test_size,
        seed=seed,
    )

    if task == "sentiment":
        train_texts = [ex["text"] for ex in train_examples]
        word2idx = build_vocab_from_texts(train_texts, max_vocab_size=max_vocab_size)

        label2idx, idx2label = build_label_mappings(data)

        train_dataset = SentimentDataset(train_examples, word2idx, label2idx)
        val_dataset = SentimentDataset(val_examples, word2idx, label2idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=sentiment_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=sentiment_collate_fn,
        )

        metadata = {
            "word2idx": word2idx,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "num_outputs": len(label2idx),
        }

    elif task == "ner":
        word2idx = build_vocab_from_ner_examples(
            train_examples,
            max_vocab_size=max_vocab_size,
        )
        tag2idx, idx2tag = build_tag_mappings()

        train_dataset = NERDataset(train_examples, word2idx, tag2idx)
        val_dataset = NERDataset(val_examples, word2idx, tag2idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ner_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ner_collate_fn,
        )

        metadata = {
            "word2idx": word2idx,
            "tag2idx": tag2idx,
            "idx2tag": idx2tag,
            "num_outputs": len(tag2idx),
        }

    else:
        raise ValueError(f"Unsupported task: {task}")

    return train_loader, val_loader, metadata


def build_model(
    task: str,
    model_name: str,
    vocab_size: int,
    num_outputs: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
):
    if task == "sentiment":
        if model_name == "meanpool":
            model = MODEL_REGISTRY["meanpool"](
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_classes=num_outputs,
                dropout=dropout,
            )
            real_model_name = "meanpool"

        elif model_name == "cnn":
            model = MODEL_REGISTRY["cnn"](
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_classes=num_outputs,
                dropout=dropout,
            )
            real_model_name = "cnn"

        elif model_name == "bilstm":
            model = MODEL_REGISTRY["bilstm"](
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_classes=num_outputs,
                num_layers=num_layers,
                dropout=dropout,
            )
            real_model_name = "bilstm"

        else:
            raise ValueError(
                f"Unsupported sentiment model_name='{model_name}'. "
                f"Available: meanpool, cnn, bilstm"
            )

    elif task == "ner":
        model = MODEL_REGISTRY["bilstm_ner"](
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_tags=num_outputs,
            num_layers=num_layers,
            dropout=dropout,
        )
        real_model_name = "bilstm_ner"

    else:
        raise ValueError(f"Unsupported task: {task}")

    return model, real_model_name


def calculate_loss_and_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    task: str,
    pad_tag_idx: int = 0,
):
    if task == "sentiment":
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
        return loss, correct, total

    if task == "ner":
        # Soporta tanto CrossEntropyLoss estándar como una loss custom
        # que recibe logits [B, T, C] y labels [B, T].
        try:
            loss = criterion(logits, labels)
        except (TypeError, RuntimeError):
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        preds = logits.argmax(dim=-1)
        mask = labels != pad_tag_idx
        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        return loss, correct, total

    raise ValueError(f"Unsupported task: {task}")


def get_metric_name(task: str) -> str:
    return "Acc" if task == "sentiment" else "TokenAcc"


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    epoch: int,
    device: torch.device,
    task: str,
    pad_tag_idx: int = 0,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_batches = 0

    for batch in train_data:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        lengths = batch[2].to(device)

        optimizer.zero_grad()

        logits = model(inputs, lengths)
        loss, batch_correct, batch_total = calculate_loss_and_counts(
            logits=logits,
            labels=labels,
            criterion=criterion,
            task=task,
            pad_tag_idx=pad_tag_idx,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_correct += batch_correct
        total_items += batch_total
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    mean_metric = total_correct / max(total_items, 1)

    if writer is not None:
        writer.add_scalar("Loss/train", mean_loss, epoch)
        writer.add_scalar(f"{get_metric_name(task)}/train", mean_metric, epoch)

    return float(mean_loss), float(mean_metric)


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    criterion: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: Optional[SummaryWriter],
    epoch: int,
    device: torch.device,
    task: str,
    pad_tag_idx: int = 0,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_batches = 0

    for batch in val_data:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        lengths = batch[2].to(device)

        logits = model(inputs, lengths)
        loss, batch_correct, batch_total = calculate_loss_and_counts(
            logits=logits,
            labels=labels,
            criterion=criterion,
            task=task,
            pad_tag_idx=pad_tag_idx,
        )

        total_loss += loss.item()
        total_correct += batch_correct
        total_items += batch_total
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    mean_metric = total_correct / max(total_items, 1)

    if writer is not None:
        writer.add_scalar("Loss/val", mean_loss, epoch)
        writer.add_scalar(f"{get_metric_name(task)}/val", mean_metric, epoch)

    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(mean_loss)
        else:
            scheduler.step()

    return float(mean_loss), float(mean_metric)


@torch.no_grad()
def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    task: str,
    pad_tag_idx: int = 0,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_batches = 0

    for batch in test_data:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        lengths = batch[2].to(device)

        logits = model(inputs, lengths)
        loss, batch_correct, batch_total = calculate_loss_and_counts(
            logits=logits,
            labels=labels,
            criterion=criterion,
            task=task,
            pad_tag_idx=pad_tag_idx,
        )

        total_loss += loss.item()
        total_correct += batch_correct
        total_items += batch_total
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    mean_metric = total_correct / max(total_items, 1)

    return float(mean_loss), float(mean_metric)