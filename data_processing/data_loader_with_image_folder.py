from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
airplane -> a.jpg, b.jpg
car -> c.jpg, d.jpg

X -> Y(labels)

a.jpg -> 0
b.jpg ->0
c.jpg -> 1
d.jpg -> 1
"""


train_dataset_path = "../data/version2/train"
test_dataset_path = "../data/version2/test"

train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = datasets.ImageFolder(train_dataset_path, train_transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

for data in train_loader:
    print("train dataset")
    images, labels = data
    print(("batch of image shape", images.shape))
    print("labels", labels)
    break


test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_dataset = datasets.ImageFolder(test_dataset_path, test_transforms)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

for data in test_loader:
    print("test dataset")
    images, labels = data
    print(("batch of image shape", images.shape))
    print("labels", labels)
    break


