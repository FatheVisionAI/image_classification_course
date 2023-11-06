import torchvision.models as models
import torch

def inception_v3(num_classes):
    model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT')
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    return model

if __name__ == '__main__':
    num_classes = 7
    model = inception_v3(num_classes)
    print(model)