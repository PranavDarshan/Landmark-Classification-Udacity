import torch
import torch.nn as nn

# Define the CNN architecture
class MyModel(nn.Module):
    """
    Defines a Convolutional Neural Network (CNN) architecture for image classification.

    Args:
        num_classes (int, optional): The number of output classes. Defaults to 1000.
        dropout (float, optional): The dropout rate for regularization. Defaults to 0.7.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super(MyModel, self).__init__()

        # Define the convolutional layers
        self.convBlock = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # layer 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        # Define the fully connected layers
        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes an input tensor through the defined CNN architecture.

        Args:
            x (torch.Tensor): The input tensor of images.

        Returns:
            torch.Tensor: The output tensor of logits or probabilities.
        """
        x = self.convBlock(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)  
        x = self.fc_layer(x)

        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=2)

def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)
    out = model(images)

    assert isinstance(out, torch.Tensor), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"
    assert out.shape == torch.Size([2, 23]), f"Expected an output tensor of size (2, 23), got {out.shape}"