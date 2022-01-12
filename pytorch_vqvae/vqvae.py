from datetime import datetime

import torch
import torch.nn.functional as F
import wandb
from tensorboardX import SummaryWriter
from torchvision import transforms, datasets
from torchvision.utils import make_grid

from pytorch_vqvae.datasets import MiniImagenet
from pytorch_vqvae.modules import VectorQuantizedVAE


def train(data_loader, model, optimizer, args, writer):
    for images, _ in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)
        wandb.log({"loss/train/reconstruction": loss_recons.item()})
        wandb.log({"loss/train/vq": loss_vq.item()})
        optimizer.step()
        args.steps += 1


def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)
    wandb.log({"loss/test/reconstruction": loss_recons.item()})
    wandb.log({"loss/test/vq": loss_vq.item()})

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde


def main(args):
    run_name = datetime.now().strftime("train-%Y-%m-%d-%H-%M")

    run = wandb.init(
        project=f"vq_{args.dataset}",
        entity='cmap_vq',
        config=None,
        name=f"{run_name}_{args.output_folder}",
    )

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}'.format(args.output_folder)
    transform_3 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:

        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                                           download=True, transform=transform_1)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                                          transform=transform_1)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                                                  train=True, download=True, transform=transform_1)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                                                 train=False, transform=transform_1)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder,
                                             train=True, download=True, transform=transform_3)
            test_dataset = datasets.CIFAR10(args.data_folder,
                                            train=False, transform=transform_3)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True,
                                     download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True,
                                     download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True,
                                    download=True, transform=transform)
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size, shuffle=False, drop_last=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)
    wandb.log({"original": wandb.Image(fixed_grid)})

    pad = 2 if args.dataset in ["fashion-mnist", "mnist"] else 1
    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k, pad).to(args.device)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction', grid, 0)

    best_loss = -1.
    for epoch in range(args.num_epochs):
        print(f"epoch {epoch}")
        train(train_loader, model, optimizer, args, writer)
        loss, _ = test(valid_loader, model, args, writer)

        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        # wandb.log({"reconstructions": wandb.Image(grid)})
        if epoch % args.log_interval == 0:
            wandb.log({"reconstruction": wandb.Image(grid)})
        writer.add_image('reconstruction', grid, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        # with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
        #    torch.save(model.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
                        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
                        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
                        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda, default: cpu)')

    parser.add_argument('--log_interval', type=int, default=1,
                        help='interval of log')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to the vqvae checkpoint ')
    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('../logs'):
        os.makedirs('../logs')
    if not os.path.exists('../models'):
        os.makedirs('../models')
    # Device
    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
