import torchvision.models as models
import torch

def resnet_50(num_classes):
    model = models.resnet50(weights = 'ResNet50_Weights.DEFAULT')
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    return model

if __name__ == '__main__':
    num_classes = 7
    model = resnet_50(num_classes)
    print(model)