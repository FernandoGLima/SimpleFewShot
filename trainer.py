import torch
import time

class Trainer():
    def __init__(self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = next(model.parameters()).device

    def train(self, manager, epochs: int, episodes: int, validate_every: int):
        train_logs = []
        val_logs = []

        self.model.train()
        start_time = time.time()

        for epoch in range(epochs):
            total_loss, total_acc = 0, 0
            train_time = time.time()

            # training
            for episode in range(episodes):
                loss, acc = self.train_step(manager)
                total_loss += loss
                total_acc += acc
                print(f'Episode {episode + 1}/{episodes}', end='\r')

            loss = total_loss / episodes
            acc = total_acc / episodes
            print(f"Epoch {epoch + 1} - Loss: {loss:.3f} - Acc: {acc:.2f} - Time: {time.time() - train_time:.0f}s")
            train_logs.append((epoch + 1, loss, acc))

            # validation
            if (epoch + 1) % validate_every == 0:
                val_time = time.time()
                val_loss, val_acc = self.validate(manager)
                print(f"Validation - Loss: {val_loss:.3f} - Acc: {val_acc:.2f} - Time: {time.time() - val_time:.0f}s\n")
                val_logs.append((epoch + 1, val_loss, val_acc))
        
        print(f'Training completed in {time.time() - start_time:.0f}s')
        return train_logs, val_logs

    def train_step(self, data_manager):
        self.model.train()

        train_loader, test_loader, _ = data_manager.get_eval_task(train_classes=True)
        loss = 0
        
        for train_batch, query_batch in zip(train_loader, test_loader):
            # this loop iterates only once because len(train_loader) == len(test_loader) == 1
            train_imgs, train_labels = train_batch
            query_imgs, query_labels = query_batch

            train_imgs, train_labels = train_imgs.to(self.device), train_labels.to(self.device)
            query_imgs, query_labels = query_imgs.to(self.device), query_labels.to(self.device)
            
            self.optimizer.zero_grad()

            if self.model.__class__.__name__ == 'FEAT':
                scores, reg = self.model(train_imgs, train_labels, query_imgs, query_labels)
                loss = self.criterion(scores, query_labels.argmax(1)) 

                all_labels = torch.cat([train_labels, query_labels], dim=0)
                loss += self.model.temperature * self.criterion(reg, all_labels)
            else:
                scores = self.model(train_imgs, train_labels, query_imgs)
                loss = self.criterion(scores, query_labels.argmax(1))
            
            l2_reg = sum(torch.norm(param) for param in self.model.parameters())
            loss += 1e-4 * l2_reg

            loss.backward()
            self.optimizer.step()

            acc = (scores.argmax(1) == query_labels.argmax(1)).float().mean()

        return loss.item(), acc.item()

    def validate(self, data_manager, episodes: int = 400):
        self.model.eval()

        n_correct = 0
        total_loss = 0

        with torch.no_grad():
            for episode in range(episodes):
                train_loader, test_loader, _ = data_manager.get_eval_task(train_classes=False)

                for train_batch, test_batch in zip(train_loader, test_loader):
                    # this loop iterates only once because len(train_loader) == len(test_loader) == 1
                    train_imgs, train_labels = train_batch
                    test_imgs, test_labels = test_batch

                    train_imgs, train_labels = train_imgs.to(self.device), train_labels.to(self.device)
                    test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)

                    scores = self.model(train_imgs, train_labels, test_imgs, test_labels)
                    if isinstance(scores, tuple): # if using FEAT ignore regularization 
                        scores = scores[0]

                    total_loss += self.criterion(scores, test_labels.argmax(1))
                    l2_reg = sum(torch.norm(param) for param in self.model.parameters())
                    total_loss += 1e-4 * l2_reg

                    n_correct += (scores.argmax(1) == test_labels.argmax(1)).sum().item()
                
                print(f'Validation {episode + 1}/{episodes}', end='\r')

        n_total = len(test_labels) * episodes
        acc = n_correct / n_total
        total_loss = total_loss / episodes

        return total_loss, acc



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