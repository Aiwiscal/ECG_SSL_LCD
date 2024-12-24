import torch
import torch.nn as nn
from models.net_base_network import EncoderNet


class MultiBranchNet(nn.Module):
    def __init__(self, num_classes, checkpoint=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(MultiBranchNet, self).__init__()
        load_params = dict()
        self.dim = 512
        encoder = EncoderNet(name="vgg", in_ch=1, mlp_hidden_size=self.dim, projection_size=self.dim, alpha=0.125)
        if checkpoint is not None:
            load_params = torch.load(checkpoint)["online_network_state_dict"]
            encoder.load_state_dict(load_params)

        encoder_ii = torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        print("lead-ii - successfully load weights.")
        self.encoder_ii = encoder_ii
        del encoder_ii

        encoder_iii = torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        print("lead-iii - successfully load weights.")
        self.encoder_iii = encoder_iii
        del encoder_iii

        encoder_v1 = torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        print("lead-v1 - successfully load weights.")
        self.encoder_v1 = encoder_v1
        del encoder_v1

        encoder_v2 = torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        print("lead-v2 - successfully load weights.")
        self.encoder_v2 = encoder_v2
        del encoder_v2

        encoder_v3 = torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        print("lead-v3 - successfully load weights.")
        self.encoder_v3 = encoder_v3
        del encoder_v3

        encoder_v4 = torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        print("lead-v4 - successfully load weights.")
        self.encoder_v4 = encoder_v4
        del encoder_v4

        encoder_v5 = torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        print("lead-v5 - successfully load weights.")
        self.encoder_v5 = encoder_v5
        del encoder_v5

        encoder_v6 = torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device)
        print("lead-v6 - successfully load weights.")
        self.encoder_v6 = encoder_v6
        del encoder_v6

        self.num_classes = num_classes
        self.num_leads = 8
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x_ii = x[:, [0], :]
        feat_ii = self.encoder_ii(x_ii)
        x_iii = x[:, [1], :]
        feat_iii = self.encoder_iii(x_iii)
        x_v1 = x[:, [2], :]
        feat_v1 = self.encoder_v1(x_v1)
        x_v2 = x[:, [3], :]
        feat_v2 = self.encoder_v2(x_v2)
        x_v3 = x[:, [4], :]
        feat_v3 = self.encoder_v3(x_v3)
        x_v4 = x[:, [5], :]
        feat_v4 = self.encoder_v4(x_v4)
        x_v5 = x[:, [6], :]
        feat_v5 = self.encoder_v5(x_v5)
        x_v6 = x[:, [7], :]
        feat_v6 = self.encoder_v6(x_v6)
        feat = torch.concat([feat_ii, feat_iii, feat_v1, feat_v2, feat_v3, feat_v4, feat_v5, feat_v6], dim=1)
        feat_ = feat.view(feat.shape[0], feat.shape[1])
        y = self.fc(feat_)
        return y
