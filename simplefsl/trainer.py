import torch
import time
from typing import Tuple, List


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        l2_weight: float = 1e-4,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.l2_weight = l2_weight
        self.device = next(model.parameters()).device
        
        method = self.model.__class__.__name__
        self.evaluate = self._evaluate if method != 'FEAT' else self._evaluate_contrastive

    def train(self,
        manager,
        epochs: int,
        episodes: int,
        validate_every: int,
    ) -> Tuple[List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
        
        start_time = time.time()
        self.model.train()
        train_logs, val_logs = [], []

        for epoch in range(epochs):
            epoch_start = time.time()
            loss, acc = self._run_episodes(manager, episodes, train=True)
            train_logs.append((epoch + 1, loss, acc))
            print(f"Epoch {epoch + 1} - Loss: {loss:.3f} - Acc: {acc:.2f} - Time: {time.time() - epoch_start:.0f}s")

            if (epoch + 1) % validate_every == 0:
                val_start = time.time()
                val_loss, val_acc = self.validate(manager)
                val_logs.append((epoch + 1, val_loss, val_acc))
                print(f"Validation - Loss: {val_loss:.3f} - Acc: {val_acc:.2f} - Time: {time.time() - val_start:.0f}s\n")

        print(f'Training completed in {time.time() - start_time:.0f}s')
        return train_logs, val_logs

    def _run_episodes(self, manager, episodes: int, train: bool) -> Tuple[float, float]:
        total_loss, total_acc = 0, 0
        self.model.train(train)

        for episode in range(episodes):
            task_data = manager.get_eval_task(train_classes=train)
            loss, acc = self._train_step(task_data) if train else self._val_step(task_data)
            total_loss += loss
            total_acc += acc
            print(f'{"Training" if train else "Validation"} {episode + 1}/{episodes}', end='\r')

        return total_loss / episodes, total_acc / episodes

    def _train_step(self, task_data) -> Tuple[float, float]:
        train_loader, test_loader, _ = task_data
        (train_imgs, train_labels), (query_imgs, query_labels) = next(zip(train_loader, test_loader))
        
        train_imgs, train_labels = train_imgs.to(self.device), train_labels.to(self.device)
        query_imgs, query_labels = query_imgs.to(self.device), query_labels.to(self.device)

        self.optimizer.zero_grad()

        loss, scores = self.evaluate(train_imgs, train_labels, query_imgs, query_labels)

        loss.backward()
        self.optimizer.step()

        acc = (scores.argmax(1) == query_labels.argmax(1)).float().mean()
        return loss.item(), acc.item()

    def _val_step(self, task_data) -> Tuple[float, float]:
        train_loader, test_loader, _ = task_data
        (train_imgs, train_labels), (test_imgs, test_labels) = next(zip(train_loader, test_loader))

        train_imgs, train_labels = train_imgs.to(self.device), train_labels.to(self.device)
        test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)

        with torch.no_grad():
            loss, scores = self.evaluate(train_imgs, train_labels, test_imgs, test_labels)
            acc = (scores.argmax(1) == test_labels.argmax(1)).float().mean()

        return loss.item(), acc.item()

    def validate(self, manager, episodes: int = 400) -> Tuple[float, float]:
        self.model.eval()
        return self._run_episodes(manager, episodes, train=False)

    def _evaluate(self,
        train_imgs: torch.Tensor,
        train_labels: torch.Tensor,
        query_imgs: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        scores = self.model(train_imgs, train_labels, query_imgs, query_labels)
        loss = self.criterion(scores, query_labels.argmax(1))
        if self.l2_weight:
            loss += self.l2_weight * sum(torch.norm(p) for p in self.model.parameters())

        return loss, scores

    def _evaluate_contrastive(self,
        train_imgs: torch.Tensor,
        train_labels: torch.Tensor,
        query_imgs: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        scores, reg = self.model(train_imgs, train_labels, query_imgs, query_labels)
        loss = self.criterion(scores, query_labels.argmax(1))
        loss += self.model.temperature * self.criterion(reg, torch.cat([train_labels, query_labels]))
        if self.l2_weight:
            loss += self.l2_weight * sum(torch.norm(p) for p in self.model.parameters())

        return loss, scores



def train_metaopt_step(data_manager, embedding_net, cls_head, optimizer, criterion, device):
    train_loader, test_loader, _ = data_manager.get_eval_task(train_classes=True)

    for train_batch, query_batch in zip(train_loader, test_loader):
        train_imgs, train_labels = train_batch
        query_imgs, query_labels = query_batch

        # remove one-hot encoding das labels
        train_labels = train_labels.argmax(dim=1).reshape(-1)
        query_labels = query_labels.argmax(dim=1).reshape(-1)

        train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
        query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)

        emb_support = embedding_net(train_imgs)
        emb_query = embedding_net(query_imgs)

        scores = cls_head(
            emb_query.unsqueeze(0),
            emb_support.unsqueeze(0),
            train_labels,
            n_way=data_manager.n_ways,
            n_shot=data_manager.n_shots,
        )

        scores = scores.squeeze(0)

        l2_reg = sum(torch.norm(param) for param in embedding_net.parameters()) + \
                 sum(torch.norm(param) for param in cls_head.parameters())        
        loss = criterion(scores, query_labels) + 1e-4 * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (scores.argmax(1) == query_labels).float().mean().item()

    return loss.item(), acc



def validate_metaopt(data_manager, embedding_net, cls_head, criterion, device, episodes=400):
    n_correct = 0
    n_total = 0
    total_loss = 0
    
    embedding_net.eval()
    cls_head.eval()
    with torch.no_grad():
        for episode in range(episodes):
            # pega uma task com classes de TESTE
            train_loader, test_loader, _ = data_manager.get_eval_task(train_classes=False)

            for train_batch, test_batch in zip(train_loader, test_loader):
                train_imgs, train_labels = train_batch
                test_imgs, test_labels = test_batch

                # remove one-hot encoding das labels
                train_labels = train_labels.argmax(dim=1).reshape(-1)
                test_labels = test_labels.argmax(dim=1).reshape(-1)

                train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
                test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

                emb_support = embedding_net(train_imgs)
                emb_test = embedding_net(test_imgs)

                scores = cls_head(
                    emb_test.unsqueeze(0),
                    emb_support.unsqueeze(0),
                    train_labels,
                    n_way=data_manager.n_ways,
                    n_shot=data_manager.n_shots,
                )

                scores = scores.squeeze(0)

                l2_reg = sum(torch.norm(param) for param in embedding_net.parameters()) + \
                    sum(torch.norm(param) for param in cls_head.parameters())
                total_loss += criterion(scores, test_labels) + 1e-4 * l2_reg

                n_correct += (scores.argmax(1) == test_labels).sum().item()
                n_total += len(test_labels)
            
            print(f'Validando {episode + 1}/{episodes}', end='\r')

    total_loss = total_loss / episodes
    acc = n_correct / n_total

    return total_loss, acc


def train_step(manager, model, optimizer, criterion, device):
    train_loader, test_loader, _ = manager.get_eval_task(train_classes=True)

    for train_batch, query_batch in zip(train_loader, test_loader):
        # esse loop so roda 1 vez (len(train_loader) == 1)
        train_imgs, train_labels = train_batch
        query_imgs, query_labels = query_batch

        # remove one-hot encoding das labels
        query_labels = query_labels.argmax(1)

        train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
        query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)

        scores = model(train_imgs, train_labels, query_imgs)
        l2_reg = sum(torch.norm(param) for param in model.parameters())
        loss = criterion(scores, query_labels) + 1e-4 * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (scores.argmax(1) == query_labels).float().mean().item()

    return loss.item(), acc

def validate(manager, model, criterion, device, episodes=400):
    n_correct = 0
    n_total = 0
    total_loss = 0

    model.eval()
    for episode in range(episodes):
        # pega uma task com classes de TESTE
        train_loader, test_loader, _ = manager.get_eval_task(train_classes=False)

        for train_batch, test_batch in zip(train_loader, test_loader):
            train_imgs, train_labels = train_batch
            test_imgs, test_labels = test_batch

            # remove one-hot encoding das labels
            test_labels = test_labels.argmax(1)

            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

            scores = model(train_imgs, train_labels, test_imgs)

            l2_reg = sum(torch.norm(param) for param in model.parameters())
            total_loss += criterion(scores, test_labels) + 1e-4 * l2_reg

            n_correct += (scores.argmax(1) == test_labels).sum().item()
            n_total += len(test_labels)
        
        print(f'Validando {episode + 1}/{episodes}', end='\r')

    total_loss = total_loss / episodes
    acc = n_correct / n_total

    return total_loss, acc