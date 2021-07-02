from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from net_model import  *

from net_model import *
from train_train import BATCH_SIZE, loss_fn, DEVICE

data_transform = transforms.Compose([
    Resize(256),
    transforms.CenterCrop(256),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.5, 0.5, 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] )
])

vali_dataset = datasets.ImageFolder(root="dataset/validation", transform=data_transform)
model = Net()
model = torch.load('model_21.pth')
model = model.to(DEVICE)
vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(model)

def predict(model, vali_loader, total_vali_loss,  total_accuracy):
    model.eval()
    with torch.no_grad():
        for data in vali_loader:
            data, target = data
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            loss = loss_fn(output, target)
            total_vali_loss = total_vali_loss + loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss: {}".format(total_vali_loss))
        print("整体测试集上的正确率: {}".format(total_accuracy / len(vali_dataset)))

predict(model, vali_loader, 0, 0)