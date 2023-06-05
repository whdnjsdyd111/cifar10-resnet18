import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_num', type=int, default=3, help='model number')
parser.add_argument('--total_epoch', type=int, default=50, help='epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batches_train', type=int, default=128, help='batch size train')
parser.add_argument('--batches_eval', type=int, default=100, help='batch size eval')

args = parser.parse_args()

model_num = args.model_num # total number of models
total_epoch = args.total_epoch # total epoch
lr = args.lr # initial learning rate
batch_size_train = args.batches_train
batch_size_eval = args.batches_eval


class Resnet18(nn.Module):
    def __init__(self, pretrained_model, dr_rate):
        super(Resnet18, self).__init__()
        self.resnet = pretrained_model
        self.dropout = nn.Dropout(p=dr_rate)


    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)

        return x


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the data transforms
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(crop)) for crop in crops]))
])
# Load the CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_eval, shuffle=False, num_workers=2)

# Define the list of models for ensemble
models = []
for i in range(model_num):
    # Define the ResNet-18 model with pre-trained weights
    pretrained_model = timm.create_model('resnet18', num_classes=10)
    model = Resnet18(pretrained_model, 0.2)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load('./resnet18_cifar10/epochs-%d/lr-%.4f/batch-%d/model_num-%d-%d.pth' % (total_epoch, lr, batch_size_train, model_num, i)))
    model.eval()  # Set the model to evaluation mode
    models.append(model)
    print('epochs-%d lr-%.4f batch-%d model_num-%d-%d load!' % (total_epoch, lr, batch_size_train, model_num, i))

# Evaluate the ensemble of models
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
        bs, ncrops, c, h, w = images.size()       
        outputs = torch.zeros(bs, 10).to(device)  # Initialize the output tensor with zeros
        for model in models:
            model_output = model(images.view(-1, c, h, w))  # Reshape the input to (bs*10, c, h, w)
            model_output = model_output.view(bs, ncrops, -1).mean(1)  # Average the predictions of the 10 crops
            outputs += model_output
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the ensemble on the 10000 test images: %f %%' % (100 * correct / total))

with open('results.txt', 'a') as f:
    f.write('\nepochs-%d, lr-%.4f, batch-%d, model_num-%d results: %f' % (total_epoch, lr, batch_size_train, model_num, (100 * correct / total)))
