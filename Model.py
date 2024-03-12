import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as models

class ResNet50Classifier(nn.Module):
    def __init__(
            self,
            ch,
            num_class,
            use_cuda,
    ):
        super(ResNet50Classifier, self).__init__()
        self.ch = ch
        self.num_class = num_class
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = models.resnet50(pretrained=True).to(device="cuda:0")
            self.model.train(mode=True)
        else:
            self.model = models.resnet50(pretrained=True).to(device="cpu")
            self.model.train(mode=True)
        self.model.conv1 = nn.Conv2d(self.ch, 64,
                                kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_extractor = self.model

        self.hidden1 = nn.Linear(1000, 300)
        nn.init.kaiming_normal_(self.hidden1.weight)
        self.bn1 = nn.BatchNorm1d(300)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.final_layer = nn.Linear(300, num_class)
        nn.init.kaiming_normal_(self.final_layer.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        extracted_features = self.feature_extractor(x)
        x = torch.nn.functional.relu(self.hidden1(extracted_features))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.final_layer(x)
        #x = torch.nn.functional.sigmoid(x)
        return x
    
    
class InceptionV3(nn.Module):
    def __init__(
            self,
            ch,
            num_class,
            use_cuda,
    ):
        super(InceptionV3, self).__init__()
        self.ch = ch
        self.num_class = num_class
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = models.inception_v3(pretrained=True).to(device="cuda:0")
            self.model.train(mode=True)
        else:
            self.model = models.inception_v3(pretrained=True).to(device="cpu")
            self.model.train(mode=True)
        self.model.Conv2d_1a_3x3 =nn.Conv2d(self.ch, 32, kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.model.Conv2d_1a_3x3.weight)
        self.feature_extractor = self.model

        self.hidden1 = nn.Linear(1000, 300)
        nn.init.kaiming_normal_(self.hidden1.weight)
        self.bn1 = nn.BatchNorm1d(300)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.final_layer = nn.Linear(300, num_class)
        nn.init.kaiming_normal_(self.final_layer.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        extracted_features = self.feature_extractor(x)
        x = torch.nn.functional.relu(self.hidden1(extracted_features))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.final_layer(x)
        #x = torch.nn.functional.sigmoid(x)
        return x

class DenseNet121(nn.Module):
    def __init__(
            self,
            ch,
            num_class,
            use_cuda,
    ):
        super(DenseNet121, self).__init__()
        self.ch = ch
        self.num_class = num_class
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = models.densenet121(pretrained=True).to(device="cuda:0")
            self.model.train(mode=True)
        else:
            self.model = models.densenet121(pretrained=True).to(device="cpu")
            self.model.train(mode=True)
        self.model.conv1 = nn.Conv2d(self.ch, 64,
                                kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_extractor = self.model

        self.hidden1 = nn.Linear(1000, 300)
        nn.init.kaiming_normal_(self.hidden1.weight)
        self.bn1 = nn.BatchNorm1d(300)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.final_layer = nn.Linear(300, num_class)
        nn.init.kaiming_normal_(self.final_layer.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        extracted_features = self.feature_extractor(x)
        x = torch.nn.functional.relu(self.hidden1(extracted_features))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.final_layer(x)
        #x = torch.nn.functional.sigmoid(x)
        return x





