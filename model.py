import torch.nn as nn
import torchvision.models as models

class ViolenceClassifier(nn.Module):
    def __init__(self):
        super(ViolenceClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 2)  # Binary classification

    def forward(self, x):
        return self.base_model(x)

if __name__ == "__main__":
    model = ViolenceClassifier()
    print(model)
