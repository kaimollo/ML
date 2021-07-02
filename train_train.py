import torch
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Resize
from net_model import *

# 需要的参数以及设备


train_data = "dataset/train"
test_data = "dataset/test"
BATCH_SIZE = 16
EPOCHS = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


writer = SummaryWriter("logs_train")

# 使用dataset加载划分好的数据集
# 使用transform处理原始图片
data_transform = transforms.Compose([
    Resize(256),
    transforms.CenterCrop(256),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.5, 0.5, 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] )
])

train_dataset = datasets.ImageFolder(root=train_data, transform=data_transform)
test_dataset = datasets.ImageFolder(root=test_data, transform=data_transform)




# 实例化模型并且移动到GPU
model = Net()
if torch.cuda.is_available():
    model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 选择Adam优化器和学习速率递减
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 训练函数

def train(model, train_loader, optimizer, epoch, scheduler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        # 优化器
        loss.backward()

        optimizer.step()



        if batch_idx % 750 == 749:
            print("训练批数：{}, train_Loss: {}".format(batch_idx+1, loss.item()))
            writer.add_scalar("train_loss", loss.item(), epoch)
    scheduler.step()


# 测试函数
total_test_loss = 0

total_accuracy = 0

def test(model, test_loader, total_test_loss,  total_accuracy, epoch):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data, target = data
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            loss = loss_fn(output, target)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的正确率: {}".format(total_accuracy / len(test_dataset)))
        writer.add_scalar("test_loss", total_test_loss, epoch)
        writer.add_scalar("test_accuracy", total_accuracy / len(test_dataset), epoch)


        torch.save(model, "model_{}.pth".format(epoch+1))
        print("模型已保存")



def train_train(EPOCHS, train_dataset, test_dataset,  BATCH_SIZE, model, scheduler, writer, optimizer):
        for epoch in range(0, EPOCHS ):
            # 导入数据并打乱
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            print("-------第 {} 轮训练开始-------".format(epoch + 1))
            train(model, train_loader, optimizer, epoch, scheduler)
            test(model, test_loader, 0,  0, epoch)


        writer.close()
