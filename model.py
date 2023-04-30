import torch
import torch.nn as nn

class ConvNN(nn.Module):
    def __init__(self, input_shape=3, hidden_units=5, output_shape=100):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        '''
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=4,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )'''
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1280,
                      out_features=output_shape)
        )
        self.activation = nn.Softmax(dim=1)
    
    def forward(self, input):
        x = self.conv_block_1(input)
        # x = self.conv_block_2(x)
        # x = self.conv_block_3(x)
        x = self.classifer(x)
        x = self.activation(x)
        return x
