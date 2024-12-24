import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class PreTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, max_epochs, gamma, batch_size,
                 num_workers):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = max_epochs
        self.writer = SummaryWriter()
        self.m = 0.996
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bn = torch.nn.BatchNorm1d(online_network.projection_size * 2, affine=False).to(device)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def cov_loss(self, x, y):
        batch_size = self.batch_size
        # empirical cross-correlation matrix
        xy = torch.concat([x, y], dim=-1)
        c = self.bn(xy).T @ self.bn(xy)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        print("\nStart PreTraining......\n")
        min_loss = 99999999
        for epoch_counter in range(self.max_epochs):
            print("\nEpoch : ", epoch_counter, "----------------------------------\n")
            total_loss = 0.0
            total_loss_intra_mse = 0.0
            total_loss_inter_mse = 0.0
            total_loss_inter_cov = 0.0
            for (batch_view_1, batch_view_2, batch_view_3), _ in tqdm(train_loader):
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                batch_view_3 = batch_view_3.to(self.device)

                loss, loss_intra_mse, loss_inter_mse, loss_inter_cov = self.update(batch_view_1, batch_view_2,
                                                                                   batch_view_3)
                self.writer.add_scalar('loss', loss, global_step=niter)
                total_loss += loss.item()
                total_loss_intra_mse += loss_intra_mse.item()
                total_loss_inter_mse += loss_inter_mse.item()
                total_loss_inter_cov += loss_inter_cov.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1
            print("\nLoss: %f - Loss_intra_mse: %f - Loss_inter_mse: %f - Loss_inter_cov: %f" % (
                total_loss / len(train_loader), total_loss_intra_mse / len(train_loader),
                total_loss_inter_mse / len(train_loader), total_loss_inter_cov / len(train_loader)))

            current_loss = total_loss / len(train_loader)
            if current_loss < min_loss:
                print("Saved at loss = %f.\n" % current_loss)
                min_loss = current_loss
                if not os.path.exists(model_checkpoints_folder):
                    os.mkdir(model_checkpoints_folder)
                self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, x1, x2, x3):
        num_leads = x1.shape[1]
        select_leads = np.random.choice(num_leads, 2, replace=False)

        h1 = self.online_network(x1[:, [select_leads[0]], :])
        h2 = self.online_network(x2[:, [select_leads[1]], :])
        q1 = self.predictor(h1)
        q2 = self.predictor(h2)

        with torch.no_grad():
            z3_ = self.target_network(x3[:, [select_leads[0]], :])

        loss_intra_mse = self.regression_loss(q1, z3_).mean()
        loss_inter_mse = self.regression_loss(q2, z3_).mean()
        loss_inter_cov = self.cov_loss(h1, h2) * self.gamma
        loss = loss_intra_mse + loss_inter_mse + loss_inter_cov
        return loss, loss_intra_mse, loss_inter_mse, loss_inter_cov

    def save_model(self, path):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
