#!/usr/bin/env python
import os
import csv
import time
import shutil
import datetime
from tqdm import tqdm
from enum import Enum
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from dataset import ImageNetDataset
# from resnet import ResNet
# from efficientnet import EfficientNet


# Strict class 
@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 3e-4
    checkpoint_path: Optional[str] = None
    save: bool = True


def get_dataloader(split, bs):
    # Data is not local 
    # data_path = Path(__file__).parent / 'data'
    data_path = Path("/Volumes/Training Data") / 'data'
    
    dataset = ImageNetDataset(data_path, split=split, verbose=True)
    return DataLoader(dataset, batch_size=bs, num_workers=16, pin_memory=True)

def train(config, result_path):
    # Instantiate the model
    device = torch.device(f'cpu')
    model = torchvision.models.resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.checkpoint_path:
        model.load_state_dict(torch.load(config.checkpoint_path))

    # Load the dataset
    train_loader = get_dataloader('train', config.batch_size)
    test_loader = get_dataloader('val', config.batch_size)

    # Training loop
    for epoch in range(100):
        model.train()
        train_loss = 0
        epoch_start_time = time.time()

        for inputs, labels in (pbar := tqdm(train_loader, leave=False)):
            start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            step_time = (time.time() - start_time) * 1000
            pbar.set_description(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Step Time: {step_time:.2f}ms")

        model.eval()
        test_loss = 0
        correct1, correct5, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in (pbar := tqdm(test_loader, leave=False)):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                _, top5_pred = torch.topk(outputs, 5, dim=1)
                total += labels.size(0)
                correct1 += (predicted == labels).sum().item()
                correct5 += (top5_pred.permute(1, 0) == labels).any(dim=0).sum().item()
                pbar.set_description(f"Epoch {epoch} | Test Loss: {loss.item():.4f} | "
                                     f"Top-1 Error: {100 - 100 * correct1 / total:.2f}% | "
                                     f"Top-5 Error: {100 - 100 * correct5 / total:.2f}%")

        # Print report and write results to CSV file
        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        top1_error = 100 - 100 * correct1 / total
        top5_error = 100 - 100 * correct5 / total
        epoch_duration = int(time.time() - epoch_start_time)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                f"Top-1 Error: {top1_error:.2f}% | Top-5 Error: {top5_error:.2f}% | "
                f"Duration: {datetime.timedelta(seconds=epoch_duration)}")

        # Save the model checkpoint
        if config.save:
            state_dict = {k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(state_dict, result_path / f'checkpoint_{epoch}.ckpt')


# Train Model
config = Config()
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_path = Path(__file__).parent / 'experiments' / current_time
result_path.mkdir(exist_ok=True, parents=True)
train(config, result_path)