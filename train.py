import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
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
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = datasets.ImageFolder(root=data_path, transform=train_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

random_seed = torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=random_seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

num_classes = len(dataset.classes)
print("\nClasses:", dataset.classes, end='\n\n')

# Initialize the model
model = CNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)  # weight decay for regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
epochs = 100

for epoch in range(epochs):
    model.train()
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
    
    scheduler.step(avg_loss)
    
    # Print accuracy and loss for each epoch
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss : {avg_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")
    print("-" * 50)
    
# Save model
torch.save(model.state_dict(), "face_model.pth")
print("\nTraining complete! Model saved as face_model.pth")