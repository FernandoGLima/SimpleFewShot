import torch

def train_step(manager, model, optimizer, criterion, device):
    train_loader, test_loader, _ = manager.get_eval_task(train_classes=True)

    for train_batch, query_batch in zip(train_loader, test_loader):
        # esse loop so roda 1 vez (len(train_loader) == 1)
        train_imgs, train_labels = train_batch
        query_imgs, query_labels = query_batch

        # remove one-hot encoding das labels
        train_labels = train_labels.argmax(1)
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
            train_labels = train_labels.argmax(1)
            test_labels = test_labels.argmax(1)

            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

            scores = model(train_imgs, train_labels, test_imgs)

            l2_reg = sum(torch.norm(param) for param in model.parameters())
            loss = criterion(scores, test_labels) + 1e-4 * l2_reg
            total_loss += loss.item()

            n_correct += (scores.argmax(1) == test_labels).sum().item()
            n_total += len(test_labels)
        
        print(f'Validando {episode + 1}/{episodes}', end='\r')

    total_loss = total_loss / episodes
    acc = n_correct / n_total

    return total_loss, acc
