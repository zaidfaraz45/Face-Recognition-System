import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Device agnostic code
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

data_path = 'faces'

# This code applies data augmentation during training to improve model accuracy.
train_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = datasets.ImageFolder(root=data_path, transform=train_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

random_seed = torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=random_seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

num_classes = len(dataset.classes)
print("\nClasses:", dataset.classes, end='\n\n')

# CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.6),  # Add dropout for regularization
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
# Initialize the model
model = CNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
epochs = 50

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Print accuracy and loss for each epoch
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss : {avg_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")
    print("-" * 50)

    
# Save model
torch.save(model.state_dict(), "face_model.pth")
print("\nTraining complete! Model saved as face_model.pth")