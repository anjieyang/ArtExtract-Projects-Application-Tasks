from loguru import logger
import datetime
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f'./src/train_{current_time}.log'
logger.add(log_file, level="INFO", encoding="utf-8")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: %s' % device)

# Setting hyperparameters
num_classes = 27
num_epochs = 5
batch_size = 512
learning_rate = 1e-3
learning_rate_decay = 0.9
fine_tune = True
pretrained=True
reg=0.001

logger.info("# ********************************** Cute Dividing Line ********************************************")
logger.info('Data Preprocessing...')
def load_dataset():
    data_aug_transforms = transforms.Compose([transforms.RandomCrop(500, pad_if_needed=True), transforms.ToTensor()])
    data_path = '/wikiart'
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=data_aug_transforms
    )
    length = len(dataset)
    num_training= int(0.6 * length)
    num_validation = int(0.2 * length)
    num_test = length - num_training - num_validation
    lengths = [num_training, num_validation, num_test] 

    # Splitting into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)
    logger.info("Training Set Size: %s"%(len(train_dataset)))
    logger.info("Validation set size: %s"%(len(val_dataset)))
    logger.info("Test set size: %s"%(len(test_dataset)))

    # Loading data using DataLoader method in torch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader, val_loader, test_loader, test_dataset


train_loader, val_loader, test_loader, test_dataset = load_dataset()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

logger.info("# ********************************** Cute Dividing Line ********************************************")
logger.info('Model Building...')
class ResNetLSTMModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(ResNetLSTMModel, self).__init__()
        
        resnet = models.resnet101(pretrained)
        set_parameter_requires_grad(resnet, fine_tune)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity()  # Removing the last fully connected layer
        self.resnet = resnet
        
        self.lstm = nn.LSTM(input_size=num_ftrs, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, n_class)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

model= ResNetLSTMModel(num_classes, fine_tune, pretrained)
# Parallelizing on multiple GPUs
model = nn.DataParallel(model)
print(model)
model.to(device)
params_to_update = model.parameters()

logger.info("# ********************************** Cute Dividing Line ********************************************")
logger.info('Model Training...')
# Updating leanrning rate during training
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Setting loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=reg)

best_accuracy = 0
ResN_Val_Acc = []
ResN_Losses = []
lr = learning_rate
total_step = len(train_loader)

# Training Process
for epoch in range(num_epochs):
    model.train()
    train_preds = []
    train_labels = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        ResN_Losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # Calculating evalution metrics Acc and F1 score
    train_acc = (torch.tensor(train_preds) == torch.tensor(train_labels)).float().mean().item()
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    print(f'Train Accuracy: {train_acc * 100:.2f}%, Train F1: {train_f1:.4f}')
    logger.info(f'Train Accuracy: {train_acc * 100:.2f}%, Train F1: {train_f1:.4f}\n')

    lr *= learning_rate_decay
    update_lr(optimizer, lr)

    logger.info("# ********************************** Cute Dividing Line ********************************************")
    logger.info('Model Validation...')
    # Validation Process
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print('Validation accuracy for {} images is: {:.2f} %, Validation F1: {:.4f}'.format(total, 100 * val_acc, val_f1))
        logger.info('Validation accuracy for {} images is: {:.2f} %, Validation F1: {:.4f}\n'.format(total, 100 * val_acc, val_f1))

logger.info("# ********************************** Cute Dividing Line ********************************************")
logger.info('Model Testing...')
# Testing Process
model.eval()
with torch.no_grad():
    test_preds = []
    test_labels = []
    correct = 0
    total = 0
    predicted_all=[]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        predicted_all.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
          break
    
    test_acc = correct / total
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    print('Accuracy of the final network on {} test images: {:.2f} %, Test F1: {:.4f}'.format(total, 100 * test_acc, test_f1))
    logger.info('Accuracy of the final network on {} test images: {:.2f} %, Test F1: {:.4f}\n'.format(total, 100 * test_acc, test_f1))
