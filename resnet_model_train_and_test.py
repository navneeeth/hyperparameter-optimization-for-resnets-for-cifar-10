
import matplotlib.pyplot as plt
import torchvision
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
import os 


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock, [8, 6, 4, 3])

class ResNetParams:
   """
    A class to pass the hyperparameters to the model
   """
   def __init__(self, arch='Model 1' ,epochs=200, start_epoch=0, batch_size=128, lr=0.1, momentum=0.9, weight_decay=1e-4, print_freq=50,
                save_dir='save_temporary_checkpoints', save_every=10):
        self.save_every = save_every #Saves checkpoints at every specified number of epochs
        self.save_dir = save_dir #The directory used to save the trained models
        self.print_freq = print_freq #print frequency 
        self.weight_decay = weight_decay #Weight decay for SGD
        self.momentum = momentum #Momentum for SGD
        self.lr = lr #Learning Rate
        self.batch_size = batch_size #Batch Size for each epoch 
        self.start_epoch = start_epoch #Starting Epoch
        self.epochs = epochs #Total Epochs
        self.arch = arch #ResNet model name

def run_epochs():
    global args, best_precision
    #Check if the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #Loading the model 
    model = project1_model()
    model.cuda()

    #Defining the Loss Function
    loss_func = nn.CrossEntropyLoss().cuda()

    #Defining the Optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    #Defining the Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    for epoch in range(args.start_epoch, args.epochs):
        #Train for one epoch
        print('Training model: {}'.format(args.arch))
        print('Current Learning Rate {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, loss_func, optimizer, epoch)
        lr_scheduler.step()

        #Test for one epoch
        precision = validate(val_loader, model, loss_func)

        #Save the best precision and make a checkpoint
        is_best = precision > best_precision
        best_precision = max(precision, best_precision)
        if epoch > 0 and epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'project1_model_checkpoint.th'))
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'project1_model.th'))
    return best_precision

class KeepAverages(object):
    #Computes and stores the average along with the current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    #Computes the top 1 precision
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, model, loss_func):
    #Run an Evaluation
    batch_time = KeepAverages()
    losses = KeepAverages()
    top1 = KeepAverages()

    #Switch to Evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            #Compute the output of the Model and calculate the Loss
            output = model(input_var)
            loss = loss_func(output, target_var)
            output = output.float()
            loss = loss.float()

            #Measure the Loss and Update it 
            precision = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(precision.item(), input.size(0))

            #Measure the elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    print('Test Accuracy\t  Top Precision: {top1.avg:.3f} (Error: {error:.3f} )\n'
          .format(top1=top1,error=100-top1.avg))
    val_losses.append(100-top1.avg)
    return top1.avg

def train(train_loader, model, loss_func, optimizer, epoch):
    #Run one training epoch

    batch_time = KeepAverages()
    data_time = KeepAverages()
    losses = KeepAverages()
    top1 = KeepAverages()

    #Switch to Train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure the data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        #Compute the output and the Loss
        output = model(input_var)
        loss = loss_func(output, target_var)

        #Compute the Gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.float()
        loss = loss.float()
        
        #Measure the accuracy and record the loss
        precision = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(precision.item(), input.size(0))

        #Measure the Elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: No: [{0}] Batches: [{1}/{2}]\t'
                  'Loss: {loss.val:.4f} (Average: {loss.avg:.4f})\t'
                  'Precision: {top1.val:.3f} (Average: {top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    train_losses.append(100-top1.val)

def imshow(img):
  #Function to show an image
#   %matplotlib inline
#   %config InlineBackend.figure_format = 'retina'
  img = img / 2 + 0.5     # un - Normalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))


args=ResNetParams()
val_losses = []
train_losses = []

#Load the data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 normalize,
                 ]), download=True),
                 batch_size=128, shuffle=True,
                 num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
               datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
               transforms.ToTensor(),
               normalize,
               ])),
               batch_size=128, shuffle=False,
               num_workers=4, pin_memory=True)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''  
#Obtaining some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
plt.figure(figsize=(20,10)) 

#Showing the images
imshow(torchvision.utils.make_grid(images[0:8,:,:]))
  
#Printing their labels
print(' '.join('%15s' % classes[labels[j]] for j in range(3)))

'''


#Load the device and move the model to the device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = project1_model().to(device)

#Obtain the summary of the loaded data
summary(model, (3,32,32))
best_precision = 0

#Run the epochs
best_precision = run_epochs()
print('The lowest error from model: {} after {} epochs is {error:.3f}'.format(args.arch,args.epochs, error=100-best_precision))
model_save_name = 'project1_model.pt'

#Saving the generated model and testing its loading
path = model_save_name
torch.save(model.state_dict(), path) 
model_path = path
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

#Plotting the model
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="Val")
plt.plot(train_losses,label="Train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('plot_graph.png')
