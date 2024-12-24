import torch
from models.mlp_head import MLPHead
from models.vgg_1d import VGG16


class EncoderNet(torch.nn.Module):
    def __init__(self, name, mlp_hidden_size, projection_size, in_ch=8, alpha=0.5):
        super(EncoderNet, self).__init__()
        net = None
        if name == "vgg":
            net = VGG16(ch_in=in_ch, n_classes=10, alpha=alpha)
            self.encoder = list(net.children())[0]
        else:
            print("unknown net : ", name)
            exit(1)
        # self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(in_channels=net.fc.in_features, mlp_hidden_size=mlp_hidden_size,
                                  projection_size=projection_size)
        self.projection_size = projection_size

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)
