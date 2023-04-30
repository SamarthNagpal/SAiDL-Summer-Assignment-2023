import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import model
import train
import metrics
import os
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

train_data = datasets.CIFAR100(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor(), target_transform=None)
test_data = datasets.CIFAR100(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor(), target_transform=None)

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


def save_results(act_func, line):
    os.chdir(r'C:\Users\samar\OneDrive\Desktop\SAIDL Project\results')
    with open(act_func + '.txt', 'a') as file:
        file.write(line)

activation_func = 'softmax'

model_0 = model.ConvNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 3
for epoch in range(epochs):
    print(f'---------EPOCH: {epoch+1}----------')
    save_results(activation_func, f'---------EPOCH: {epoch+1}----------\n')
    train_loss = 0
    model_0.train()
    for batch, (x_train, y_train) in tqdm(enumerate(train_dataloader)):
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        y_logits = model_0(x_train)
        y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, y_train)
        train_loss += loss
        loss.backward()
        optimizer.step()
    print(f'training loss: {loss/len(train_dataloader)}')
    save_results(activation_func, f'training loss: {loss/len(train_dataloader)}\n')
    model_0.eval()
    with torch.inference_mode():
        # for x_test, y_test in test_dataloader:
        #     test_preds = model_0(x_test)
        #     accuracy = metrics.accuracy(test_preds, y_test)
        #     precision = metrics.precision(test_preds, y_test)
        #     recall = metrics.recall(test_preds, y_test)
        #     # f1score = metrics.f1score(test_preds, y_test)
        x_test, y_test = torch.from_numpy(test_data.data).type(torch.float32), torch.Tensor(test_data.targets).type(torch.float32)
        y_test = y_test.type(torch.LongTensor)
        x_test, y_test = x_test.to(device), y_test.to(device)
        x_test = x_test.permute(0, 3, 1, 2)
        test_preds = model_0(x_test)
        print(f'test loss: {loss_fn(test_preds, y_test)}')
        accuracy = metrics.accuracy(test_preds.argmax(dim=1), y_test)
        precision = metrics.precision(test_preds.argmax(dim=1), y_test)
        recall = metrics.recall(test_preds.argmax(dim=1), y_test)
        f1score = metrics.f1score(test_preds.argmax(dim=1), y_test)
        print(f'acc: {accuracy}, precision: {precision}, recall: {recall}, f1score: {f1score}')
        save_results(activation_func, f'test loss: {loss_fn(test_preds, y_test)}, acc: {accuracy}, precision: {precision}, recall: {recall}, f1score: {f1score}\n')

    print('----------EPOCH DONE-------------')
    save_results(activation_func, '----------EPOCH DONE-------------\n')

save_results(activation_func, '\n')