# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20

import pdb
import argparse
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms

from torch.utils.data.dataloader import RandomSampler
from util.misc import CSVLogger
from util.cutout import Cutout

from model.resnet import ResNet18
from model.wide_resnet import WideResNet

model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model

print(args)

# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
else:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)
elif args.dataset == 'svhn':
    num_classes = 10
    train_dataset = datasets.SVHN(root='data/',
                                  split='train',
                                  transform=train_transform,
                                  download=True)

    extra_dataset = datasets.SVHN(root='data/',
                                  split='extra',
                                  transform=train_transform,
                                  download=True)

    # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
    data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
    labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
    train_dataset.data = data
    train_dataset.labels = labels

    test_dataset = datasets.SVHN(root='data/',
                                 split='test',
                                 transform=test_transform,
                                 download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                        #    sampler=RandomSampler(train_dataset, True, 40000),
                                           pin_memory=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=0)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

cnn = cnn.cuda()

criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

filename = 'logs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc', 'xentropy', 'var', 'avg_var', 'arg_var', 'index', 'labels'], filename=filename)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc

kl_sum = 0
y_bar = torch.zeros(8, 10).detach().cuda()

# y_bar 구하는 epoch
for epoch in range(1):
    # checkpoint = torch.load('C:/Users/82109/Desktop/캡디/캡디자료들/논문모델/cutout/checkpoints/sampling/sampling_{0}.pt'.format(8), map_location = torch.device('cuda:0'))
    checkpoint = torch.load('C:/Users/82109/Desktop/캡디/캡디자료들/논문모델/cutout/checkpoints/1_cifar10_resnet18.pt', map_location = torch.device('cuda:0'))
    cnn.load_state_dict(checkpoint)
    cnn.eval()
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    norm_const = 0

    kldiv = 0
    # pred_sum = torch.Tensor([0] * 10).detach().cuda()
    count = 0
    label_list = []
    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()
        label_list.append(labels.item())
        save_image(images[0], os.path.join('C:/Users/82109/Desktop/캡디/캡디자료들/논문모델/cutout/augmented_images/{0}/'.format(0), 'img{0}.png'.format(i)))

        cnn.zero_grad()
        pred = cnn(images)
        xentropy_loss = criterion(pred, labels)
        # xentropy_loss.backward()
        # cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        pred_softmax = nn.functional.softmax(pred).cuda()
        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total
        for a in range(pred_softmax.data.size()[0]):
            for b in range(y_bar.size()[1]):
                y_bar[epoch][b] += torch.log(pred_softmax.data[a][b])
        

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
        count += 1
    xentropy = xentropy_loss_avg / count
    y_bar[epoch] = torch.Tensor([x / 50000 for x in y_bar[epoch]]).cuda()
    y_bar[epoch] = torch.exp(y_bar[epoch])
    for index in range(y_bar.size()[1]):
        norm_const += y_bar[epoch][index]
    for index in range(y_bar.size()[1]):
        y_bar[epoch][index] = y_bar[epoch][index] / norm_const
    print("y_bar[{0}] : ".format(epoch), y_bar[epoch])
    test_acc = test(test_loader)
    # print(pred, labels.data)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step(epoch)  # Use this line for PyTorch <1.4
    # scheduler.step()     # Use this line for PyTorch >=1.4

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc), 'xentropy' : float(xentropy)
    }
    csv_logger.writerow(row)
    # del pred
    # torch.cuda.empty_cache()


var_tensor = torch.zeros(8, 50000).detach().cuda()
var_addeachcol = torch.zeros(1, 50000).detach().cuda()

# kl_div 구하는 epoch
for epoch in range(1):
    # checkpoint = torch.load('C:/Users/82109/Desktop/캡디/캡디자료들/논문모델/cutout/checkpoints/sampling/sampling_{0}.pt'.format(8), map_location = torch.device('cuda:0'))
    checkpoint = torch.load('C:/Users/82109/Desktop/캡디/캡디자료들/논문모델/cutout/checkpoints/1_cifar10_resnet18.pt', map_location = torch.device('cuda:0'))
    cnn.load_state_dict(checkpoint)
    cnn.eval()
    kldiv = 0
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch) + ': Calculate kl_div')
        
        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        pred = cnn(images)

        pred_softmax = nn.functional.softmax(pred).cuda()

        # 입력 두 개의 shape이 다르면 batchsize로 평균을 내서 반환.
        kldiv = torch.nn.functional.kl_div(y_bar[epoch], pred_softmax, reduction='sum')
        # 1 * 50000에 한 모델의 데이터별 variance 저장
        var_tensor[epoch][i] += abs(kldiv).detach()
        var_addeachcol[0][i] += var_tensor[epoch][i]
        kl_sum += kldiv.detach()
        # print(y_bar_copy.size(), pred_softmax.size())
        # print(kl_sum)
    var = abs(kl_sum.item() / 50000)
    print("Variance : ", var)
    csv_logger.writerow({'var' : float(var)})
    # print(var_tensor)
for i in range(var_addeachcol.size()[1]):
    var_addeachcol[0][i] = var_addeachcol[0][i] / 8

print(var_addeachcol)
# var_addeachcol[0] = torch.Tensor([x / 8 for x in var_addeachcol]).cuda()
var_sorted = torch.argsort(var_addeachcol)
print(var_sorted)
for i in range(var_addeachcol.size()[1]):
    csv_logger.writerow({'avg_var' : float(var_addeachcol[0][i]), 'arg_var' : float(var_sorted[0][i]), 'index' : float(i + 1), 'labels' : float(label_list[i])})
torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
