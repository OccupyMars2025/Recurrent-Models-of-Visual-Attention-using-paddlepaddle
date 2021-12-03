import numpy as np
import paddle.io

from utils import plot_images

# import torch
# from torchvision import datasets
# from torchvision import transforms
from paddle.vision import transforms, datasets
# from torch.utils.data.sampler import SubsetRandomSampler



def get_train_valid_loader(
    # data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    # pin_memory=False,
):
    """Train and validation data loaders.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle: whether to shuffle the train/validation indices.
        show_sample: plot 9x9 sample grid of the dataset.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    # dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)
    dataset = datasets.MNIST(mode='train', download=True, transform=trans)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # train_idx, valid_idx = indices[split:], indices[:split]

    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # train_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     sampler=train_sampler,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    # )

    train_loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    # valid_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     sampler=valid_sampler,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    # )

    valid_loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    # visualize some images
    if show_sample:
        # sample_loader = torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=9,
        #     shuffle=shuffle,
        #     num_workers=num_workers,
        #     pin_memory=pin_memory,
        # )
        sample_loader = paddle.io.DataLoader(
            dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


# def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
def get_test_loader(batch_size, num_workers=4):

    """Test datalaoder.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    # dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
    dataset = datasets.MNIST(mode='test', download=True, transform=trans)

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    # )
    data_loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return data_loader


if __name__ == "__main__":
    paddle.disable_static()

    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = datasets.MNIST(mode='train', download=True, transform=trans)
    print(dataset)
    print("len(dataset): ", len(dataset))
    # for i in dataset:
    #     print(i)
    #     exit(22)
    train_loader, valid_loader = get_train_valid_loader(
                                    # data_dir,
                                    batch_size=9,
                                    random_seed=20,
                                    valid_size=0.1,
                                    shuffle=True,
                                    show_sample=False,
                                    num_workers=1,
                                    # pin_memory=False,
    )
    print("len(train_loader.dataset), len(valid_loader.dataset): ",
          len(train_loader.dataset), len(valid_loader.dataset))
    # for x, y in train_loader:
    #     print(x, y, sep=2*'\n')
    #     break