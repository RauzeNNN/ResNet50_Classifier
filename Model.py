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
            dropout_rate,
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
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
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
    
    
class MobileNetClassifier(nn.Module):
    def __init__(
            self,
            ch,
            num_class,
            use_cuda,
            dropout_rate,
    ):
        super(MobileNetClassifier, self).__init__()
        self.ch = ch
        self.num_class = num_class
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = models.mobilenet_v3_small(pretrained=True).to(device="cuda:0")
            self.model.train(mode=True)
        else:
            self.model = models.mobilenet_v3_small(pretrained=True).to(device="cpu")
            self.model.train(mode=True)

        
        ######change first convolution
        # Get the original first convolution layer
        original_conv = self.model.features[0][0]

        # Create a new convolution layer with 1 input channel
        new_conv = nn.Conv2d(
            in_channels=1,  # For grayscale images
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )

        # Initialize the weights of the new convolution layer
        # by averaging the weights of the original layer
        new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # Replace the original convolution layer with the new one
        self.model.features[0][0] = new_conv

        self.feature_extractor = self.model
        self.hidden1 = nn.Linear(1000, 300)
        nn.init.kaiming_normal_(self.hidden1.weight)
        self.bn1 = nn.BatchNorm1d(300)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
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
    