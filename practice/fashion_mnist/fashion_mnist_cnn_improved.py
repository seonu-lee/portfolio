"""
Fashion MNIST 이미지 분류 성능 개선
다중 로지스틱 회귀 → CNN으로 업그레이드

기존 모델의 한계:
1. 공간적 구조 정보 손실 (784차원 벡터로 펼침)
2. 선형 결정 경계의 한계
3. 특징 추출 능력 부재

개선 방향:
1. CNN을 통한 공간적 특징 학습
2. 비선형 활성화 함수 사용
3. 계층적 특징 추출
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 시각화 스타일 설정
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# ====================================
# 1. 데이터 로딩 (기존과 동일)
# ====================================
print("=" * 60)
print("데이터 로딩 시작...")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

print(f"✓ 학습 데이터: {len(trainset)}개")
print(f"✓ 테스트 데이터: {len(testset)}개\n")

# 클래스 이름 매핑
class_names = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}


# ====================================
# 2. CNN 모델 정의 - 3가지 버전
# ====================================

# 모델 1: 기본 CNN (Simple CNN)
class SimpleCNN(nn.Module):
    """
    간단한 CNN 구조
    - Conv layers: 특징 추출
    - Pooling: 차원 축소
    - FC layers: 분류
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 합성곱 레이어 1: 1채널 → 32채널
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 합성곱 레이어 2: 32채널 → 64채널
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # MaxPooling: 2x2로 크기 절반
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout: 과적합 방지
        self.dropout = nn.Dropout(0.25)
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv1 → ReLU → Pool: (28, 28) → (14, 14)
        x = self.pool(F.relu(self.conv1(x)))
        # Conv2 → ReLU → Pool: (14, 14) → (7, 7)
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten: (64, 7, 7) → (64*7*7)
        x = x.view(-1, 64 * 7 * 7)
        # FC1 → ReLU → Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # FC2 (출력층)
        x = self.fc2(x)
        return x


# 모델 2: 개선된 CNN (Improved CNN)
class ImprovedCNN(nn.Module):
    """
    개선된 CNN 구조
    - 더 깊은 네트워크 (3개의 Conv block)
    - Batch Normalization 추가
    - 더 많은 필터
    """
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # FC Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Block 1: (28, 28) → (14, 14)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Block 2: (14, 14) → (7, 7)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # Block 3: (7, 7) → (3, 3)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        # Flatten & FC
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# 모델 3: 고급 CNN (Advanced CNN with Residual Connection)
class AdvancedCNN(nn.Module):
    """
    고급 CNN 구조
    - Residual connection (skip connection)
    - 더 깊은 네트워크
    - Global Average Pooling
    """
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        # Initial Conv
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual Block 1
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.shortcut1 = nn.Conv2d(32, 64, kernel_size=1)
        
        # Residual Block 2
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Global Average Pooling 후 FC
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        # Initial Conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # (28, 28) → (14, 14)
        
        # Residual Block 1
        identity = self.shortcut1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity)  # Skip connection
        x = self.pool(x)  # (14, 14) → (7, 7)
        x = self.dropout(x)
        
        # Residual Block 2
        identity = self.shortcut2(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = F.relu(x + identity)  # Skip connection
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ====================================
# 3. 학습 함수
# ====================================
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, model_name="Model"):
    """
    모델 학습 및 평가
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name} 학습 시작")
    print(f"{'=' * 60}")
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}\n")
    model = model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 학습률 스케줄러 (학습 진행에 따라 학습률 감소)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 기록용
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 통계
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 에폭별 학습 정확도
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # 테스트 정확도 평가
        test_acc = evaluate_model(model, test_loader, device, verbose=False)
        
        # 기록
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")
        
        # 학습률 스케줄러 업데이트
        scheduler.step()
    
    print(f"\n✓ {model_name} 학습 완료!")
    print(f"최종 테스트 정확도: {test_accs[-1]:.2f}%\n")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'final_acc': test_accs[-1]
    }


# ====================================
# 4. 평가 함수
# ====================================
def evaluate_model(model, test_loader, device, verbose=True):
    """
    모델 평가
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    
    if verbose:
        print(f"전체 정확도: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy


# ====================================
# 5. 오분류 분석 함수
# ====================================
def analyze_misclassifications(model, test_loader, device, num_samples=10):
    """
    오분류 샘플 추출 및 분석
    """
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # 틀린 샘플 찾기
            mask = predicted != labels
            if mask.any():
                wrong_images = images[mask]
                wrong_preds = predicted[mask]
                wrong_labels = labels[mask]
                wrong_probs = probs[mask]
                
                for i in range(len(wrong_images)):
                    if len(misclassified) >= num_samples:
                        break
                    
                    misclassified.append({
                        'image': wrong_images[i].cpu(),
                        'predicted': wrong_preds[i].item(),
                        'true': wrong_labels[i].item(),
                        'confidence': wrong_probs[i][wrong_preds[i]].item()
                    })
            
            if len(misclassified) >= num_samples:
                break
    
    return misclassified


def visualize_misclassifications(misclassified, model_name):
    """
    오분류 샘플 시각화
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, sample in enumerate(misclassified[:10]):
        img = sample['image'].squeeze()
        pred = sample['predicted']
        true = sample['true']
        conf = sample['confidence']
        
        axes[i].imshow(img, cmap='binary')
        axes[i].set_title(
            f"예측: {class_names[pred]}\n"
            f"정답: {class_names[true]}\n"
            f"확신도: {conf:.2f}",
            fontsize=9
        )
        axes[i].axis('off')
    
    plt.suptitle(f"{model_name} - 오분류 샘플 분석", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ====================================
# 6. 클래스별 정확도 분석
# ====================================
def class_wise_accuracy(model, test_loader, device):
    """
    클래스별 정확도 계산
    """
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    class_accs = {}
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        class_accs[class_names[i]] = acc
    
    return class_accs


def visualize_class_accuracy(class_accs, model_name):
    """
    클래스별 정확도 시각화
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(class_accs.keys())
    accs = list(class_accs.values())
    
    colors = ['#FF6B6B' if acc < 85 else '#4ECDC4' if acc < 90 else '#95E1D3' for acc in accs]
    bars = ax.barh(classes, accs, color=colors)
    
    # 값 표시
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 0.5, i, f'{acc:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('정확도 (%)', fontsize=12)
    ax.set_title(f'{model_name} - 클래스별 정확도', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ====================================
# 7. 메인 실행
# ====================================
if __name__ == "__main__":
    # 재현성을 위한 시드 설정
    torch.manual_seed(42)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델들 정의
    models = {
        'SimpleCNN': SimpleCNN(),
        'ImprovedCNN': ImprovedCNN(),
        'AdvancedCNN': AdvancedCNN()
    }
    
    # 결과 저장
    results = {}
    
    print("\n" + "="*60)
    print("Fashion MNIST 분류 - CNN 기반 성능 개선")
    print("="*60)
    print("\n기존 모델:")
    print("  - z_model (nn.Linear): 83.53%")
    print("  - SoftmaxModel (Custom): 78.60%")
    print("\n목표: CNN을 통한 90%+ 정확도 달성\n")
    
    # 각 모델 학습
    for name, model in models.items():
        results[name] = train_model(
            model, train_loader, test_loader,
            epochs=15, lr=0.001, model_name=name
        )
    
    # ====================================
    # 8. 결과 비교 및 시각화
    # ====================================
    print("\n" + "="*60)
    print("결과 요약")
    print("="*60)
    print(f"{'모델':<20} {'최종 정확도':<15} {'개선율':<15}")
    print("-"*60)
    
    baseline = 83.53  # 기존 z_model 정확도
    for name, result in results.items():
        acc = result['final_acc']
        improvement = acc - baseline
        print(f"{name:<20} {acc:>6.2f}%         {improvement:>+6.2f}%")
    
    # 학습 곡선 비교
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    for name, result in results.items():
        axes[0].plot(result['train_losses'], label=name, marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('학습 Loss 비교')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Train Accuracy
    for name, result in results.items():
        axes[1].plot(result['train_accs'], label=name, marker='o')
    axes[1].axhline(y=baseline, color='r', linestyle='--', label='기존 모델 (83.53%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('학습 정확도 비교')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Test Accuracy
    for name, result in results.items():
        axes[2].plot(result['test_accs'], label=name, marker='o')
    axes[2].axhline(y=baseline, color='r', linestyle='--', label='기존 모델 (83.53%)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('테스트 정확도 비교')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ 학습 곡선 저장: training_comparison.png")
    
    # 최고 성능 모델로 상세 분석
    best_model_name = max(results, key=lambda x: results[x]['final_acc'])
    best_model = models[best_model_name]
    
    print(f"\n최고 성능 모델: {best_model_name} ({results[best_model_name]['final_acc']:.2f}%)")
    print("상세 분석 진행 중...\n")
    
    # 오분류 분석
    misclassified = analyze_misclassifications(best_model, test_loader, device, num_samples=10)
    fig_mis = visualize_misclassifications(misclassified, best_model_name)
    plt.savefig('results/misclassifications.png', dpi=300, bbox_inches='tight')
    print("✓ 오분류 샘플 저장: misclassifications.png")
    
    # 클래스별 정확도
    class_accs = class_wise_accuracy(best_model, test_loader, device)
    fig_class = visualize_class_accuracy(class_accs, best_model_name)
    plt.savefig('results/class_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ 클래스별 정확도 저장: class_accuracy.png")
    
    # 클래스별 정확도 출력
    print("\n" + "="*60)
    print("클래스별 정확도")
    print("="*60)
    for class_name, acc in sorted(class_accs.items(), key=lambda x: x[1]):
        print(f"{class_name:<15} {acc:>6.2f}%")
    
    # 모델 저장
    torch.save(best_model.state_dict(), 'results/best_model.pth')
    print(f"\n 최고 성능 모델 저장: best_model.pth")
    
    print("\n" + "="*60)
    print("모든 작업 완료!")
    print("="*60)
    print("\n생성된 파일:")
    print("  1. training_comparison.png - 모델별 학습 곡선 비교")
    print("  2. misclassifications.png - 오분류 샘플 분석")
    print("  3. class_accuracy.png - 클래스별 정확도")
    print("  4. best_model.pth - 최고 성능 모델 가중치")
