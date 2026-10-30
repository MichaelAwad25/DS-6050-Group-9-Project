import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class_list = ["N", "S", "V", "F", "Q"]


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.sequential(x)


class CompactCNN(nn.Module):

    def __init__(self, num_classes=5, dropout_prob=0.2):
        
        super(CompactCNN, self).__init__()

        layers = []

        layers.append(ConvBlock(1, 32, kernel_size=11))

        layers.append(ConvBlock(32, 64, kernel_size=5))

        layers.append(ConvBlock(64, 128, kernel_size=3))

        self.sequential = nn.Sequential(*layers)

        self.average_pooling = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout_prob)

        self.linear_transformation = nn.Linear(128, num_classes)


    def forward(self, x):
        intermediate = self.sequential(x)
        intermediate = self.average_pooling(intermediate)

        intermediate = torch.flatten(intermediate, start_dim=1)
        intermediate = self.dropout(intermediate)

        return self.linear_transformation(intermediate)



class BeatDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        window = torch.tensor(self.windows[i], dtype=torch.float32).unsqueeze(0)
        target = int(self.labels[i])
        return window, target


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, seed: int, batch_size: int, num_workers: int):
    train_ds = BeatDataset(X_train, y_train)
    val_ds = BeatDataset(X_val, y_val)
    test_ds = BeatDataset(X_test, y_test)
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers, generator = g, pin_memory = pin
    )
    val_loader = DataLoader(
        val_ds, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = pin
    )
    test_loader = DataLoader(
        test_ds, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = pin
    )
    return train_loader, val_loader, test_loader


def set_seeds_to(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def train_epoch(model, dataloader, criterion, optimizer, device, feature_extract = False):
    """Train the model for one epoch."""
    model.train()
    if feature_extract:
        model.apply(_set_bn_eval)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_model(device, model, train_loader, val_loader, num_classes, num_epochs=10, lr=0.001, feature_extract = False, step_size = 5, gamma = 0.1):
    """
    Train and evaluate a model.

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, feature_extract = feature_extract
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        _, _, val_macro_f1 = compute_metrics(model, val_loader, device, num_classes)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_macro_f1)

        print(f'Train Loss: {train_loss}, Train Acc: {train_acc}%')
        print(f'Val Loss: {val_loss}, Val Acc: {val_acc}%, Val Macro-F1: {val_macro_f1:.3f}')

        scheduler.step()

    return history



def plot_training_history(history, title="Training History"):
    """Plot training and validation loss/accuracy."""
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(history['val_macro_f1'], label='Val Macro-F1')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Macro-F1')
    ax3.set_title(f'{title} - Macro-F1')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"training_history_for_{title}.png")
    plt.show()



def compute_metrics(model, dataloader, device, num_classes):
    """Confusion matrix, per-class sensitivity, PPV, F1, and macro-F1."""
    model.eval()

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for true_class, predicted_class in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[true_class, predicted_class] += 1

    per_class = {}
    list_of_f1 = []
    for i, name in enumerate(class_list[:num_classes]):
        true_positives = confusion_matrix[i, i]
        support = confusion_matrix[i, :].sum()
        number_predicted = confusion_matrix[:, i].sum()

        sensitivity = true_positives / support if support > 0 else 0.0
        precision = true_positives / number_predicted if number_predicted > 0 else 0.0
        f1 = 2 * sensitivity * precision / (sensitivity + precision) if sensitivity + precision > 0 else 0.0

        per_class[name] = {'sensitivity': sensitivity, 'precision': precision, 'f1': f1, 'support': support}

        if support > 0:
            list_of_f1.append(f1)

    macro_f1 = sum(list_of_f1) / len(list_of_f1)
    return confusion_matrix, per_class, macro_f1


def print_metrics(confusion_matrix, per_class, macro_f1):
    total = confusion_matrix.sum()
    correct = sum(confusion_matrix[i, i] for i in range(len(per_class)))
    print(f"\nTest Macro-F1: {macro_f1:.3f}")
    print(f"Test Accuracy: {100.0 * correct / total:.2f}%")

    print(f"\n{'Class':6s}{'Sens':>8s}{'PPV':>8s}{'F1':>8s}{'Support':>10s}")
    for name, values in per_class.items():
        print(f"{name:6s}{values['sensitivity']:8.3f}{values['precision']:8.3f}{values['f1']:8.3f}{values['support']:10d}")

    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(f"{'':6s}" + "".join(f"{name:>8s}" for name in per_class))
    for i, name in enumerate(per_class):
        print(f"{name:6s}" + "".join(f"{confusion_matrix[i, j]:8d}" for j in range(len(per_class))))



def test_CompactCNN(CompactCNN, accelerator, train_loader, val_loader, test_loader, num_classes) -> dict:
    print("\n" + "="*60)
    print("Baseline: Compact 1D CNN")
    print("="*60)
    model = CompactCNN(num_classes=num_classes)
    print(f"CompactCNN Parameters: {sum(p.numel() for p in model.parameters()):,}")
    history = train_model(accelerator, model, train_loader, val_loader, num_classes, num_epochs=10)
    plot_training_history(history, "CompactCNN")

    confusion_matrix, per_class, macro_f1 = compute_metrics(model, test_loader, accelerator, num_classes)
    print_metrics(confusion_matrix, per_class, macro_f1)
    return history


def main() -> None:
    batch_size = 256
    num_classes = 5
    num_workers = 0
    seed = 42
    set_seeds_to(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerator {device} will be used.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Need to be filled in after data is cleaned
    # X_train, y_train = 
    # X_val, y_val = 
    # X_test, y_test = 

    test_CompactCNN(CompactCNN, device, num_classes)


if __name__ == '__main__':
    main()