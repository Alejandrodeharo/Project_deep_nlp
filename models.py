from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn


class MeanPoolClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)  # [B, T, E]
        mask = (x != self.pad_idx).unsqueeze(-1).float()  # [B, T, 1]
        emb = emb * mask
        pooled = emb.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).float()
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class CNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_idx: int = 0,
        num_filters: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        emb = self.embedding(x).transpose(1, 2)  # [B, E, T]
        conv_out = torch.relu(self.conv(emb))    # [B, F, T]
        pooled = torch.max(conv_out, dim=2).values
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int = 0,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]

        h = self.dropout(h)
        return self.fc(h)


class BiLSTMNER(nn.Module):
    """
    BiLSTM para NER.
    Devuelve logits por token: [batch, seq_len, num_tags]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.lstm(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
        )

        output = self.dropout(output)
        logits = self.fc(output)
        return logits


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "meanpool": MeanPoolClassifier,
    "cnn": CNNTextClassifier,
    "bilstm": BiLSTMClassifier,
    "bilstm_ner": BiLSTMNER,
}