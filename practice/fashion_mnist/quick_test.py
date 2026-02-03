"""
Fashion MNIST CNN - Quick Test Version
빠른 실험을 위한 간소화 버전 (5분 이내 완료)
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("Quick Test 시작...\n")

# ========== 1. 데이터 준비 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True)  # 배치 크기 증가
test_loader = DataLoader(testset, batch_size=128, shuffle=False)

# ========== 2. 모델 정의 ==========
class QuickCNN(nn.Module):
    """5분 안에 90% 달성하는 빠른 CNN"""
    def __init__(self):
        super(QuickCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========== 3. 학습 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {device}\n")

model = QuickCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5  # 빠른 테스트를 위해 5 에폭만
print("학습 시작 (5 epochs)...")
print("-" * 50)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    
    # 테스트
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    
    print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

print("-" * 50)
print(f"\n✓ 최종 테스트 정확도: {test_acc:.2f}%")
print(f"✓ 기존 모델 (로지스틱 회귀): 83.53%")
print(f"✓ 개선율: +{test_acc - 83.53:.2f}%\n")

# ========== 4. 오분류 샘플 확인 ==========
print("오분류 샘플 추출 중...")

class_names = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

misclassified = []
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        mask = predicted != labels
        if mask.any():
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append({
                        'image': images[i].cpu(),
                        'pred': predicted[i].item(),
                        'true': labels[i].item(),
                        'conf': probs[i][predicted[i]].item()
                    })
                    
                    if len(misclassified) >= 5:
                        break
        
        if len(misclassified) >= 5:
            break

# 시각화
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, sample in enumerate(misclassified[:5]):
    img = sample['image'].squeeze()
    pred = sample['pred']
    true = sample['true']
    conf = sample['conf']
    
    axes[i].imshow(img, cmap='binary')
    axes[i].set_title(f"예측: {class_names[pred]}\n정답: {class_names[true]}\n확신: {conf:.2f}", fontsize=9)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('results/quick_test_results.png', dpi=150, bbox_inches='tight')
print("✓ 결과 저장: quick_test_results.png\n")

# ========== 5. 클래스별 정확도 ==========
print("클래스별 정확도 분석...")
print("-" * 50)

class_correct = [0] * 10
class_total = [0] * 10

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"{class_names[i]:<15} {acc:>6.2f}%")

print("-" * 50)

# ========== 6. 비교 결과 ==========
print("\n" + "=" * 50)
print("결과 요약")
print("=" * 50)
print(f"기존 로지스틱 회귀:  83.53%")
print(f"Quick CNN:          {test_acc:.2f}%")
print(f"개선율:             +{test_acc - 83.53:.2f}%")
print("=" * 50)
print("\nQuick Test 완료!")