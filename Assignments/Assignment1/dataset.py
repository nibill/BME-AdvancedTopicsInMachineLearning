import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import cv2

class MnistPairs(Dataset):
    """Dataset with Mnist pairs."""

    def __init__(self, root, train, download, transform=None, order='right', return_original_labels=False):
    # def __init__(self, ds, transform=None, order='right', return_original_labels=False):  
        """
        Args:
            root (string): Directory to store the downloaded MNIST dataset.
            train (bool): If True, use the training part of the MNIST dataset.
            download(bool): If True, will download the dataset, if it is not in the root folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            order (str): Indicates which ordering of digits to use, ['right', 'left'].
            return_original_labels (bool): Indicates if it is needed to return the original MNIST labels.
        """
        
        assert order in ['right', 'left'], "Got unexpected order argument. Expected one of ['right', 'left']"
        self.train = train
        self.download = download
        self.transform = transform
        self.order = order
        
        self.return_original_labels = return_original_labels
        
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform)

    def __len__(self):
        # MnistPairs should be half the size of the MNIST dataset
        return len(self.mnist_dataset) // 2

    def __getitem__(self, idx):
        # You need to implement this method in such a way
        # that the ith element of the MnistPairs class
        # is a pair of subsequent MNIST dataset samples.
        # Make sure that you process the order in a right way.
        # That is if MNIST is [a, b, c, d], then MnistPairs
        # with the 'right' order are [[a, b], [c, d]],
        # and [[b, a], [d, c]] for the 'left' order.
        # The label is mod 10 sum of the MNIST labels.
        
        first_image = None
        first_label = None
        second_image = None
        second_label = None
        label = None
        
        #########################
        #     Your code         #
        #########################
        
        img, target = self.mnist_dataset.data[idx * 2], int(self.mnist_dataset.targets[idx * 2])

        if self.order == 'left':
            first_image = self.mnist_dataset.data[idx * 2 + 1]
            first_label = int(self.mnist_dataset.targets[idx * 2 + 1])
            second_image = self.mnist_dataset.data[idx * 2]
            second_label = int(self.mnist_dataset.targets[idx * 2])
            # first_image = Image.fromarray(img[1].numpy(), mode='L')
            # first_label = target[1]
            # second_image = Image.fromarray(img[0].numpy(), mode='L')
            # second_label = target[0]
        elif self.order == 'right':
            first_image = self.mnist_dataset.data[idx * 2]
            first_label = int(self.mnist_dataset.targets[idx * 2])
            second_image = self.mnist_dataset.data[idx * 2 + 1]
            second_label = int(self.mnist_dataset.targets[idx * 2 + 1])
            # first_image = Image.fromarray(img[0].numpy(), mode='L')
            # first_label = target[0]
            # second_image = Image.fromarray(img[1].numpy(), mode='L')
            # second_label = target[1]

        label = (first_label + second_label)
        image = cv2.hconcat([first_image.numpy(), second_image.numpy()])
        
        #########################
        #     End of your code  #
        #########################
        
        if self.return_original_labels:
            return first_image, second_image, label, first_label, second_label
        
        return image, label

        # return first_image, second_image, label
