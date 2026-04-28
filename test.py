import tqdm
import torch
import torch.nn as nn
from model import CNN
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Device agnostic code
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

data_path = 'faces'

test_transform = transforms.Compose([
    transforms.Resize((100, 100)),  
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Load dataset
dataset = datasets.ImageFolder(root=data_path, transform=test_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

random_seed = torch.manual_seed(42)
_, test_dataset = random_split(dataset, [train_size, test_size], generator=random_seed)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

num_classes = len(dataset.classes)
print("\nClasses:", dataset.classes, end='\n\n')

# Load model
model = CNN(num_classes).to(device)
model.load_state_dict(torch.load("face_model.pth"))
model.eval()

criterion = nn.CrossEntropyLoss()

# Testing
correct = 0
total = 0
total_loss = 0

progress = tqdm.tqdm(test_loader, desc="Testing", leave=True)

with torch.no_grad():
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
avg_loss = total_loss / len(test_loader)

print("=" * 40)
print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
print("=" * 40)
