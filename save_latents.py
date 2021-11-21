from datetime import datetime

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms, datasets

from datasets import MiniImagenet
from modules import VectorQuantizedVAE


def model_logits(codebook, inputs):
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.view(-1, embedding_size)

    codebook_sqr = torch.sum(codebook ** 2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

    # Compute the distances to the codebook
    distances = torch.addmm(codebook_sqr + inputs_sqr,
                            inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
    return distances


def get_latent_dataset(data_loader: iter, model: torch.nn.Module) -> TensorDataset:
    """
    Get the latent dataset over the given loader

    Parameters
    ----------
    data_loader : loads of data
    model : vae to use to get the latents

    Returns
    -------
    latents dataset
    """
    with torch.no_grad():
        features_arr = []
        labels_arr = []
        for batch, labels in data_loader:
            features = model_logits(model.codebook, batch.to(model.device))
            if type(features) == tuple:
                features = features[0]
            features_arr.append(features.to("cpu"))
            labels_arr.append(labels.to("cpu"))
        features = torch.cat(features_arr, axis=0)
        labels = torch.cat(labels_arr, axis=0)
    return TensorDataset(features, labels)


def main(args):
    output_folder = os.path.join("data", args.dataset, args.identifier)
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

    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
    model.load_state_dict(torch.load(args.vq_path, map_location=args.device)).to(args.device)
    model.eval()

    for key, loader in {"train": train_loader, "val": valid_loader, "test": test_loader}.items():
        latent_dataset = get_latent_dataset(loader, model)
        torch.save(latent_dataset, os.path.join(output_folder, f"{key}_latents.pt"))

    torch.save(model, os.path.join(output_folder, os.path.join(output_folder, "model.pt")))
    torch.save(model.state_dict(), os.path.join(output_folder, os.path.join(output_folder, "state_dict.pt")))


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
                        help='name of the data folder')
    parser.add_argument('--vq_path', type=str,
                        help='path of the vq statedict')
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

    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda, default: cpu)')
    parser.add_argument('--identifier', type=str, default=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
                        , help='name of the output folder ')
    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')

    main(args)
