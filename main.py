import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

"""
import os
from time import sleep

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

from dataset import load_data
from models.lenet import LeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#1) Loading the train and test data
#2) Loading the model 
#3) Hyper Parameters:- such as learning rate, no. of epochs, Loss and Optimizer 
#4) training loop
#5) training the training loss and accuracy
#6) saving the model checkpoint



# model accuracy check
def model_report(data, model):
    # Initialize the prediction and label lists(tensors)
    y_pred = np.zeros(0)
    y_true = np.zeros(0)
    model.eval()
    with torch.no_grad():
        for data_batch, y_true_batch in data:
            data_batch = data_batch.to(device=device)

            scores = model(data_batch)
            _, y_pred_batch = scores.max(1)
            y_pred_batch = y_pred_batch.cpu().numpy()

            y_pred = np.concatenate((y_pred, y_pred_batch))
            y_true = np.concatenate((y_true, y_true_batch))

    model.train()
    print(classification_report(y_true, y_pred,)) # target_names=labels))


# Loading the train and test data
dataset_path = "data/version1"
image_size = (32, 32)
batch_size = 16
train_data, test_data = load_data(dataset_path, image_size, batch_size)

# Loading the model
in_channel = 3
num_classes = 7
model = LeNet(in_channel=in_channel, num_classes=num_classes)
# model = torch.compile(model)
model.to(device)
model.train()

# Hyper Parameters Loss and Optimizer
learning_rate = 1e-3
num_epochs = 30
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop
start_time = time.time()
for epoch in range(1, num_epochs+1):
    pbar_batch = tqdm(train_data, unit="batch")
    losses = []
    for data in pbar_batch:
        pbar_batch.set_description(f"Epoch {epoch}")
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        loss = criterion(scores, labels)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cost = sum(losses) / len(losses)
        pbar_batch.set_postfix(loss=cost)
        sleep(0.1)

    if (epoch % 5) == 0:
        model_report(test_data, model)
        sleep(0.1)

end_time = time.time()
print(f"Total time for model training in min is {(end_time - start_time)/60}")

# saving the model checkpoint
model_save_path = "weights"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

torch.save(model.state_dict(), model_save_path + "/lenet_model_checkpoint.pth")
"""