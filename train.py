from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from train_functions import build_dataloaders, build_model, train_step, val_step
from utils import save_checkpoint, set_seed


def build_ner_class_weights(
    tag2idx: dict[str, int],
    weight_o: float = 1.0,
    weight_b: float = 2.0,
    weight_i: float = 2.5,
) -> torch.Tensor:
    """
    Construye pesos por clase para NER.

    - O pesa menos para evitar que el modelo colapse a O.
    - B-* e I-* pesan más.
    - <PAD> se ignora con peso 0.
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


def build_valid_transition_mask(tag2idx: dict[str, int]) -> torch.Tensor:
    """
    Máscara BIO de transiciones válidas entre etiquetas.

    valid[prev, curr] = 1 si la transición prev -> curr es válida.
    """
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
                # Permitimos <PAD> después de cualquier etiqueta válida.
                is_valid = True
            elif curr_tag == "O":
                is_valid = True
            elif curr_prefix == "B":
                # Un B-* puede empezar después de casi cualquier cosa.
                is_valid = True
            elif curr_prefix == "I":
                # I-X solo puede seguir a B-X o I-X.
                if prev_prefix in {"B", "I"} and prev_label == curr_label:
                    is_valid = True
                else:
                    is_valid = False
            else:
                is_valid = False

            valid[prev_idx, curr_idx] = 1.0 if is_valid else 0.0

    return valid


class NERStructureAwareLoss(nn.Module):
    """
    Loss estructurada para NER que combina:

    1. CrossEntropy ponderada por clase
    2. Penalización por omitir entidades (gold entity -> prob alta en O)
    3. Penalización por falsos positivos (gold O -> prob alta en entidad)
    4. Penalización por transiciones BIO inválidas

    Esto corrige mejor los errores observados:
    - entidades inventadas
    - spans mal cerrados
    - I-* sin B-* previo
    """

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
        """
        logits: [B, T, C]
        labels: [B, T]
        """
        num_classes = logits.size(-1)

        # 1) CE ponderada
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            ignore_index=self.pad_idx,
        )

        probs = torch.softmax(logits, dim=-1)

        valid_mask = labels != self.pad_idx
        gold_entity_mask = valid_mask & (labels != self.o_idx)
        gold_o_mask = valid_mask & (labels == self.o_idx)

        # 2) Penaliza omitir entidad real como O
        p_o = probs[..., self.o_idx]
        if gold_entity_mask.any():
            miss_penalty = -torch.log((1.0 - p_o[gold_entity_mask]).clamp(min=1e-8)).mean()
        else:
            miss_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)

        # 3) Penaliza inventar entidad donde gold es O
        p_entity = 1.0 - p_o
        if gold_o_mask.any():
            fp_penalty = -torch.log((1.0 - p_entity[gold_o_mask]).clamp(min=1e-8)).mean()
        else:
            fp_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)

        # 4) Penaliza transiciones inválidas usando probabilidades suaves
        # expected_invalid_mass = sum_{prev,curr inválidos} p(prev)*p(curr)
        if logits.size(1) > 1:
            prev_probs = probs[:, :-1, :]   # [B, T-1, C]
            curr_probs = probs[:, 1:, :]    # [B, T-1, C]

            pair_probs = prev_probs.unsqueeze(-1) * curr_probs.unsqueeze(-2)  # [B,T-1,C,C]
            invalid_mask = 1.0 - self.valid_transition_mask  # [C,C]

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
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--task", choices=["sentiment", "ner"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bilstm")
    parser.add_argument("--save_path", type=str, default="models/best_model.pt")

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_vocab_size", type=int, default=50000)

    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)

    # Hiperparámetros de la loss estructurada para NER
    parser.add_argument("--ner_lambda_miss", type=float, default=1.2)
    parser.add_argument("--ner_lambda_fp", type=float, default=1.0)
    parser.add_argument("--ner_lambda_transition", type=float, default=0.8)
    parser.add_argument("--ner_weight_o", type=float, default=1.0)
    parser.add_argument("--ner_weight_b", type=float, default=2.0)
    parser.add_argument("--ner_weight_i", type=float, default=2.5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_loader, val_loader, metadata = build_dataloaders(
        task=args.task,
        data_path=args.data_path,
        test_size=args.test_size,
        seed=args.seed,
        batch_size=args.batch_size,
        max_vocab_size=args.max_vocab_size,
    )

    model, real_model_name = build_model(
        task=args.task,
        model_name=args.model_name,
        vocab_size=len(metadata["word2idx"]),
        num_outputs=metadata["num_outputs"],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = model.to(device)

    if args.task == "sentiment":
        criterion = nn.CrossEntropyLoss()
        metric_name = "acc"
        selection_metric_name = "acc"
        pad_tag_idx = 0
    else:
        pad_tag_idx = metadata["tag2idx"]["<PAD>"]

        class_weights = build_ner_class_weights(
            metadata["tag2idx"],
            weight_o=args.ner_weight_o,
            weight_b=args.ner_weight_b,
            weight_i=args.ner_weight_i,
        ).to(device)

        criterion = NERStructureAwareLoss(
            tag2idx=metadata["tag2idx"],
            class_weights=class_weights,
            lambda_miss=args.ner_lambda_miss,
            lambda_fp=args.ner_lambda_fp,
            lambda_transition=args.ner_lambda_transition,
        ).to(device)

        # Seguimos mostrando token_acc, pero el mejor modelo se selecciona por entity_f1.
        metric_name = "token_acc"
        selection_metric_name = "entity_f1"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    run_name = (
        f"{args.task}_{real_model_name}"
        f"_lr_{args.lr}"
        f"_bs_{args.batch_size}"
        f"_ep_{args.epochs}"
    )
    writer = SummaryWriter(f"runs/{run_name}")

    best_selection_metric = -1.0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss, train_metric, train_extra_metrics = train_step(
            model=model,
            train_data=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            device=device,
            task=args.task,
            pad_tag_idx=pad_tag_idx,
        )

        val_loss, val_metric, val_extra_metrics = val_step(
            model=model,
            val_data=val_loader,
            criterion=criterion,
            scheduler=scheduler,
            writer=writer,
            epoch=epoch,
            device=device,
            task=args.task,
            pad_tag_idx=pad_tag_idx,
        )

        if args.task == "sentiment":
            current_selection_metric = val_metric
        else:
            current_selection_metric = val_extra_metrics["entity_f1"]

        if current_selection_metric > best_selection_metric:
            best_selection_metric = current_selection_metric
            save_checkpoint(
                {
                    "task": args.task,
                    "model_name": real_model_name,
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "metadata": metadata,
                },
                args.save_path,
            )

        if (epoch + 1) % args.print_every == 0 or epoch == args.epochs - 1:
            if args.task == "sentiment":
                print(
                    f"Epoch {epoch + 1}/{args.epochs}\n"
                    f"Train Loss: {train_loss:.4f}\n"
                    f"Val Loss: {val_loss:.4f}\n"
                    f"Train {metric_name}: {train_metric:.4f}\n"
                    f"Val {metric_name}: {val_metric:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{args.epochs}\n"
                    f"Train Loss: {train_loss:.4f}\n"
                    f"Val Loss: {val_loss:.4f}\n"
                    f"Train token_acc: {train_metric:.4f}\n"
                    f"Val token_acc: {val_metric:.4f}\n"
                    f"Train entity_f1: {train_extra_metrics['entity_f1']:.4f}\n"
                    f"Val entity_f1: {val_extra_metrics['entity_f1']:.4f}\n"
                    f"Train entity_precision: {train_extra_metrics['entity_precision']:.4f}\n"
                    f"Val entity_precision: {val_extra_metrics['entity_precision']:.4f}\n"
                    f"Train entity_recall: {train_extra_metrics['entity_recall']:.4f}\n"
                    f"Val entity_recall: {val_extra_metrics['entity_recall']:.4f}"
                )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best validation {selection_metric_name}: {best_selection_metric:.4f}")
    print(f"Checkpoint saved in: {args.save_path}")

    writer.close()


if __name__ == "__main__":
    main()