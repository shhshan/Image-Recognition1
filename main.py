import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary  # 用于查看模型结构

# 设备选择（优先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 数据预处理（使用 MNIST 官方均值和标准差）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义改进版 CNN（增加第二层卷积层）
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 新增第二层卷积
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 适配新的卷积层输出
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)  # 通过第二个卷积层
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)  # 展平
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 创建模型并移动到 GPU
model = ImprovedCNN().to(device)
print(summary(model, input_size=(1, 28, 28)))  # 打印模型结构

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()  # 训练模式
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # 移动到 GPU

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print(f'==> Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(train_loader):.4f}')

# 测试模型
model.eval()  # 评估模式
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # 移动到 GPU
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

# 可视化部分预测结果
num_images_shown = 0  # 控制显示的图片数量
plt.figure(figsize=(10, 5))

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)  # 迁移到 GPU
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    for j in range(images.size(0)):
        if num_images_shown < 10:  # 仅显示前 0 个样本
            plt.subplot(2, 5, num_images_shown + 1)
            plt.imshow(images[j].cpu().squeeze(), cmap='gray')
            plt.title(f'Pred: {predicted[j].item()}, True: {labels[j].item()}')
            plt.axis('off')
            num_images_shown += 1
        else:
            break
    if num_images_shown >= 10:
        break  # 退出循环

plt.show()
