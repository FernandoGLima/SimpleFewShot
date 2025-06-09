import torch
import torch.nn as nn
import time
from tqdm import tqdm

from typing import Tuple, List

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim,
        l2_weight: float = 1e-4,
    ):
        self.model = model
        self.criterion = criterion 
        self.optimizer = optimizer
        self.l2_weight = l2_weight
        self.device = next(model.parameters()).device
        
    def train(self,
        manager,
        epochs: int,
        train_episodes: int,
        validate_every: int = 2, 
        val_episodes: int = 400,
        save_checkpoint: bool = False
    ) -> Tuple[List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
        
        start_time = time.time()
        self.model.train()
        train_logs, val_logs = [], []

        best_val_acc = 0.0

        for epoch in range(epochs):
            epoch_start = time.time()
            loss, acc = self._run_episodes(manager, train_episodes, train=True)
            train_logs.append((epoch + 1, loss, acc))
            print(f"Epoch {epoch + 1} - Loss: {loss:.3f} - Acc: {acc:.2f} - Time: {time.time() - epoch_start:.0f}s")

            if validate_every > 0 and (epoch + 1) % validate_every == 0:
                val_start = time.time()
                val_loss, val_acc = self.validate(manager, val_episodes)
                val_logs.append((epoch + 1, val_loss, val_acc))
                print(f"Validation - Loss: {val_loss:.3f} - Acc: {val_acc:.2f} - Time: {time.time() - val_start:.0f}s\n")

                if save_checkpoint and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(f'./checkpoints/{self.model.__class__.__name__}{manager.ways}w{manager.shots}s.pth')

        print(f'Training completed in {time.time() - start_time:.0f}s')
        print(f'Best validation acc: {max(val_logs, key=lambda x: x[2])[2]:.2f} at epoch {max(val_logs, key=lambda x: x[2])[0]}')
        return train_logs, val_logs

    def _run_episodes(self, manager, episodes: int, train: bool) -> Tuple[float, float]:
        total_loss, total_acc = 0.0, 0.0
        self.model.train(train)

        for _ in tqdm(range(episodes), desc="Training" if train else "Validation", leave=False):
            task_data = manager.get_fewshot_task(train)
            loss, acc = self._train_step(task_data) if train else self._val_step(task_data)
            total_loss += loss
            total_acc += acc

        return total_loss / episodes, total_acc / episodes

    def _train_step(self, task_data) -> Tuple[float, float]:
        train_imgs, train_labels, query_imgs, query_labels = self._prepare_task_data(task_data)

        loss, acc = self.evaluate(train_imgs, train_labels, query_imgs, query_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), acc.item()

    def _val_step(self, task_data) -> Tuple[float, float]:
        train_imgs, train_labels, test_imgs, test_labels = self._prepare_task_data(task_data)

        # with torch.no_grad():
        loss, acc = self.evaluate(train_imgs, train_labels, test_imgs, test_labels)

        return loss.item(), acc.item()

    def validate(self, manager, episodes) -> Tuple[float, float]:
        self.model.eval()
        return self._run_episodes(manager, episodes, train=False)

    def evaluate(self,
        train_imgs: torch.Tensor,
        train_labels: torch.Tensor,
        query_imgs: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # for models with custom loss/evaluation logic
        if hasattr(self.model, 'evaluate'): 
            return self.model.evaluate(train_imgs, train_labels, query_imgs, query_labels, self.criterion, self.l2_weight)
        
        scores = self.model(train_imgs, train_labels, query_imgs)
        acc = (scores.argmax(1) == query_labels.argmax(1)).float().mean()

        query_labels = query_labels.float() if isinstance(self.criterion, torch.nn.MSELoss) else query_labels.argmax(1)
        loss = self.criterion(scores, query_labels)

        if self.l2_weight:
            loss += self.l2_weight * sum(torch.norm(p) for p in self.model.parameters())

        return loss, acc

    def _prepare_task_data(self, task_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train_loader, test_loader, _ = task_data
        (train_imgs, train_labels), (test_imgs, test_labels) = next(zip(train_loader, test_loader))

        train_imgs, train_labels = train_imgs.to(self.device), train_labels.to(self.device)
        test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
    
        return train_imgs, train_labels, test_imgs, test_labels

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f">> Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):   
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f">> Checkpoint loaded from {path}")