import argparse
from pathlib import Path
import torch
import torchvision.transforms as transforms
from data_utils.multi_view_data_injector import MultiViewDataInjector
from models.mlp_head import MLPHead
from models.net_base_network import EncoderNet
from models.pre_trainer import PreTrainer
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from data_utils.data_folder import ECGDatasetFolder

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Lead Correlation and Decorrelation Pretraining')
parser.add_argument('--data-dir', type=Path, required=True,
                    metavar='DIR', help='data path')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--gamma', default=0.005, type=float, metavar='L',
                    help='balance parameter of the loss')


def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PreTraining with: {device}")

    print("Pretraining Setting ======================================")
    print(args)
    print("==========================================================")

    t = transforms.Compose([
        RandomResizeCropTimeOut(),
        ToTensor()
    ])

    train_dataset = ECGDatasetFolder(args.data_dir,
                                     transform=MultiViewDataInjector([t, t, t]))
    mlp_dim = 512

    # online network
    online_network = EncoderNet(name="vgg", mlp_hidden_size=mlp_dim, projection_size=mlp_dim, alpha=0.125, in_ch=1).to(
        device)

    # predictor network
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        mlp_hidden_size=mlp_dim, projection_size=mlp_dim).to(device)

    # target encoder
    target_network = EncoderNet(name="vgg", mlp_hidden_size=mlp_dim, projection_size=mlp_dim, alpha=0.125, in_ch=1).to(
        device)

    optimizer = torch.optim.Adam(list(online_network.parameters()) + list(predictor.parameters()),
                                 lr=args.learning_rate)

    trainer = PreTrainer(online_network=online_network,
                         target_network=target_network,
                         optimizer=optimizer,
                         predictor=predictor,
                         device=device,
                         max_epochs=args.epochs,
                         gamma=args.gamma,
                         batch_size=args.batch_size,
                         num_workers=args.workers)

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
