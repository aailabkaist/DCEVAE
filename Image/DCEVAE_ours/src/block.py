from torch import nn

class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)

class Conv_block(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, kernel_size, stride=1, padding=0, negative_slope=0.2,
                 p=0.04, transpose=False):
        super(Conv_block, self).__init__()

        self.transpose = transpose
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        self.dropout = nn.Dropout2d(p)
        self.batch_norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.conv(x)
        if not self.transpose:
            x = self.dropout(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x