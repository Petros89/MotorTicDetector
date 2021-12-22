import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import shutil


torch.cuda.empty_cache()

np.random.seed(0)
torch.manual_seed(0)


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; max validation accuracy
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

#sns.set_style('darkgrid')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

root_dir = "Dataset"
print("The data lies here =>", root_dir)


image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225] )]),  #normalize the image
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225])])  #normalize the image
        #transforms.Normalize(1.2368e-05, 1.0000)])
    }


Tourette = datasets.ImageFolder(root = root_dir,
                                      transform = image_transforms["train"]
                                     )

Tourette.class_to_idx


idx2class = {v: k for k, v in Tourette.class_to_idx.items()}

def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    for _, label_id in dataset_obj:
        label = idx2class[label_id]
        count_dict[label] += 1
    return count_dict

def plot_from_dict(dict_obj, plot_title, **kwargs):
    return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)


plt.figure(figsize=(15,8))
plot_from_dict(get_class_distribution(Tourette), plot_title="Entire Dataset (before train, val split)")
plt.show()

Tourette_size = len(Tourette)

Tourette_indices = list(range(Tourette_size))

np.random.shuffle(Tourette_indices)

val_split_index = int(np.floor(0.2 * Tourette_size))

train_idx, val_idx = Tourette_indices[val_split_index:], Tourette_indices[:val_split_index]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)



# This is for test dataset
test_dir = "Test"
Tourette_test = datasets.ImageFolder(root = test_dir, transform = image_transforms["test"])
#test_sampler = scaler.fit_tranform(Tourette_test)

# Load data
train_loader = DataLoader(dataset=Tourette, shuffle=False, batch_size=16, sampler=train_sampler)

val_loader = DataLoader(dataset=Tourette, shuffle=False, batch_size=1, sampler=val_sampler)

test_loader = DataLoader(dataset=Tourette_test, shuffle=False, batch_size=1)


def get_class_distribution_loaders(dataloader_obj, dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    if dataloader_obj.batch_size == 1:
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else:
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
    return count_dict


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
plot_from_dict(get_class_distribution_loaders(train_loader, Tourette), plot_title="Train Set", ax=axes[0])
plot_from_dict(get_class_distribution_loaders(val_loader, Tourette), plot_title="Val Set", ax=axes[1])
plt.show()


single_batch = next(iter(train_loader))
print(single_batch[0].shape)

print("Output label tensors: ", single_batch[1])
print("\nOutput label tensor shape: ", single_batch[1].shape)


# Selecting the first image tensor from the batch.
single_image = single_batch[0][0]
print(single_image.shape)
print(single_image)

#plt.imshow(single_image(1, 2, 0))
#plt.show()

single_batch_grid = utils.make_grid(single_batch[0], nrow=4)
plt.figure(figsize = (10,10))
plt.imshow(single_batch_grid.permute(1, 2, 0))
plt.show()


### build CNN Classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=56, stride=1, padding=0)
        # kernel_size_default = 56
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block


model = CNNClassifier()
model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)
#optimizer = optim.Adam(model.parameters(), lr=0.001)



########## Training ##################
def train(start_epoch, n_epochs, val_acc_max, train_loader, val_loader, criterion, optimizer, model, checkpoint_path, best_model_path):
     
    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
        correct_results_sum = (y_pred_tags == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)
        return acc

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    val_acc_max=0
    print("Begin training.")
    for e in tqdm(range(start_epoch, n_epochs)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch).squeeze()
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = binary_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch).squeeze()
                # uncomment next line for setting batch_size = 1 to the val set`
                y_val_pred = torch.unsqueeze(y_val_pred, 0)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = binary_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()


        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': e + 1,
            'val_acc_max': val_epoch_acc/len(val_loader),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if val_epoch_acc/len(val_loader) > val_acc_max/len(val_loader):
            print('Validation accuracy increased ({:.5f} --> {:.5f}).  Saving model...'.format(val_acc_max/len(val_loader), val_epoch_acc/len(val_loader)))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            #save_ckp(checkpoint, True, "./checkpoint/current_checkpoint.pt", "./best_model/best_model_path.pt")
            val_acc_max = val_epoch_acc

    return model, accuracy_stats, loss_stats

# lets train
start = 1
end = 50
trained_model, accuracy_stats, loss_stats = train(start, end, 0.5, train_loader, val_loader, criterion, optimizer, model, "./checkpoint/current_checkpoint_test.pt", "./best_model/best_model_test.pt")

# plot training and validation history curves
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
plt.show()


# save model to the path for later use
#torch.save(model.state_dict(), MODEL_PATH)


########## Validation Set Predictions ###########################
Y_pred_list = []
Y_true_list = []

with torch.no_grad():
    for X_batch, Y_batch in tqdm(val_loader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_val_pred = model(X_batch)
        #Y_val_pred = torch.unsqueeze(Y_val_pred, 0)
        _, Y_pred_tag = torch.max(Y_val_pred, dim = 1)
        Y_pred_list.append(Y_pred_tag.cpu().numpy())
        Y_true_list.append(Y_batch.cpu().numpy())


Y_pred_list = [i[0][0][0] for i in Y_pred_list]
Y_true_list = [i[0] for i in Y_true_list]


# classification report
print(classification_report(Y_true_list, Y_pred_list))

# confusion matrix
print(confusion_matrix(Y_true_list, Y_pred_list))

# plot confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(Y_true_list, Y_pred_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(confusion_matrix_df, cmap='Blues', annot=True, ax=ax)
plt.title("Confusion Matrix [Validation Set]")
plt.show()

# plot confusion matrix in percent
confusion_matrix_df = pd.DataFrame(confusion_matrix(Y_true_list, Y_pred_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(confusion_matrix_df/np.sum(confusion_matrix_df), cmap='Blues', annot=True, fmt='.2%', ax=ax)
plt.title("Confusion Matrix [Validation Set]")
plt.show()


########## Testing Set Predictions ###########################
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())


y_pred_list = [i[0][0][0] for i in y_pred_list]
y_true_list = [i[0] for i in y_true_list]


# classification report
print(classification_report(y_true_list, y_pred_list))

# confusion matrix
print(confusion_matrix(y_true_list, y_pred_list))

# plot confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(confusion_matrix_df, annot=True, cmap='Blues', ax=ax)
plt.title("Confusion Matrix [Test Set]")
plt.show()

# plot confusion matrix in percent
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(confusion_matrix_df/np.sum(confusion_matrix_df), cmap='Blues', annot=True, fmt='.2%', ax=ax)
plt.title("Confusion Matrix [Validation Set]")
plt.show()
