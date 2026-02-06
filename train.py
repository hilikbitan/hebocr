"""Optional CRNN training entrypoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from model import CRNN


class DummyDataset(Dataset):
    """Placeholder dataset for future training integration."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError("No data available")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(img_h=args.img_height, num_channels=1, num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CTCLoss(blank=0)

    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model.train()
    for _ in range(args.epochs):
        for _batch in loader:
            optimizer.zero_grad()
            loss = criterion(torch.zeros(1), torch.zeros(1, dtype=torch.int32), torch.zeros(1), torch.zeros(1))
            loss.backward()
            optimizer.step()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional CRNN training")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--output", type=str, default="")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
