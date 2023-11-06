import torchvision.models as models
import torch

def vgg16_bn(num_classes):
    model = models.vgg16_bn(weights = 'VGG16_BN_Weights.DEFAULT')
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([torch.nn.Linear(num_features, num_classes)])
    model.classifier = torch.nn.Sequential(*features)
    return  model

if __name__ == '__main__':
    num_classes = 7
    model = vgg16_bn(num_classes)
    print(model)