from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from train_functions import build_dataloaders, build_model, train_step, val_step
from utils import save_checkpoint, set_seed


class NERSpanAwareLoss(nn.Module):
    """
    Loss para NER que penaliza más los falsos negativos de entidad.

    Combina:
    1. CrossEntropy ponderada por clase.
    2. Penalización extra cuando el gold es una entidad (B-* o I-*)
       y el modelo asigna mucha probabilidad a la clase O.

    Esto hace que "omitir" una entidad prediciendo O sea más costoso.
    """

    def __init__(
        self,
        tag2idx: dict[str, int],
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
        p_o = probs[..., self.o_idx]  # [B, T]

        valid_mask = labels != self.pad_idx
        entity_mask = valid_mask & (labels != self.o_idx)

        if entity_mask.any():
            # Si el token es realmente entidad, castigamos que el modelo
            # le dé alta probabilidad a O.
            miss_penalty = -torch.log((1.0 - p_o[entity_mask]).clamp(min=1e-8)).mean()
        else:
            miss_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)

        return ce_loss + self.lambda_miss * miss_penalty


def build_ner_class_weights(
    tag2idx: dict[str, int],
    weight_o: float = 1.0,
    weight_b: float = 2.0,
    weight_i: float = 3.0,
) -> torch.Tensor:
    """
    Construye pesos por clase para NER.

    - O pesa menos para que el modelo no tienda a colapsar hacia O.
    - B-* pesa más.
    - I-* pesa aún más, porque omitir continuidad de entidad suele ser costoso.
    - <PAD> queda en 0, ya que se ignora.
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

    # Nuevos argumentos: solo afectan a NER
    parser.add_argument(
        "--use_span_aware_ner_loss",
        action="store_true",
        help="Usa la loss custom para penalizar entidades omitidas como O.",
    )
    parser.add_argument(
        "--ner_lambda_miss",
        type=float,
        default=1.5,
        help="Peso de la penalización extra cuando una entidad real se empuja hacia O.",
    )
    parser.add_argument(
        "--ner_weight_o",
        type=float,
        default=1.0,
        help="Peso de la clase O en la CrossEntropy de NER.",
    )
    parser.add_argument(
        "--ner_weight_b",
        type=float,
        default=2.0,
        help="Peso de las clases B-* en la CrossEntropy de NER.",
    )
    parser.add_argument(
        "--ner_weight_i",
        type=float,
        default=3.0,
        help="Peso de las clases I-* en la CrossEntropy de NER.",
    )

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
        pad_tag_idx = 0
    else:
        pad_tag_idx = metadata["tag2idx"]["<PAD>"]
        metric_name = "token_acc"

        if args.use_span_aware_ner_loss:
            class_weights = build_ner_class_weights(
                metadata["tag2idx"],
                weight_o=args.ner_weight_o,
                weight_b=args.ner_weight_b,
                weight_i=args.ner_weight_i,
            ).to(device)

            criterion = NERSpanAwareLoss(
                tag2idx=metadata["tag2idx"],
                class_weights=class_weights,
                lambda_miss=args.ner_lambda_miss,
            ).to(device)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_idx)

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

    best_metric = -1.0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss, train_metric = train_step(
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

        val_loss, val_metric = val_step(
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

        if val_metric > best_metric:
            best_metric = val_metric
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
            print(
                f"Epoch {epoch + 1}/{args.epochs}\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Train {metric_name}: {train_metric:.4f}\n"
                f"Val {metric_name}: {val_metric:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best validation {metric_name}: {best_metric:.4f}")
    print(f"Checkpoint saved in: {args.save_path}")

    writer.close()


if __name__ == "__main__":
    main()