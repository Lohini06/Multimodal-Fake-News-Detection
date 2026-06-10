import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json

class Trainer:
    def __init__(self, model, config: dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"]
        )

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc="Training", leave=False)
        for images, texts, labels in loop:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images, texts)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        return total_loss / len(train_loader), correct / total

    def val_epoch(self, val_loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, texts, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images, texts)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(val_loader), correct / total

    def save_checkpoint(self, epoch: int, val_acc: float):
        path = os.path.join(
            self.config["outputs"]["checkpoint_dir"],
            self.config["outputs"]["best_model_name"]
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
        }, path)
        print("Checkpoint saved (val_acc: " + str(round(val_acc, 4)) + ")")

    def fit(self, train_loader, val_loader):
        epochs = self.config["training"]["epochs"]
        patience = self.config["training"]["early_stopping_patience"]

        print("Starting training for " + str(epochs) + " epochs...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.val_epoch(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print("Epoch " + str(epoch) + "/" + str(epochs) +
                  " | Train Loss: " + str(round(train_loss, 4)) +
                  " Acc: " + str(round(train_acc, 4)) +
                  " | Val Loss: " + str(round(val_loss, 4)) +
                  " Acc: " + str(round(val_acc, 4)))

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print("Early stopping at epoch " + str(epoch))
                    break

        history_path = os.path.join(self.config["outputs"]["results_dir"], "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print("Training complete. Best val acc: " + str(round(self.best_val_acc, 4)))
        return self.history