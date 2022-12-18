from dagster import op, multi_asset, AssetOut, Out, graph, GraphOut
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

loading_emnist = 'load_mnst'


@multi_asset(group_name=loading_emnist, outs={
    "mnist_train": AssetOut(),
    "mnist_test": AssetOut(),
})
def mnist() -> tuple[Dataset, Dataset]:
    ''''
    Load the MNIST dataset, both the training and test set
    '''
    mnist_train = MNIST('./data/', download=True, train=True, )
    mnist_test = MNIST('./data/', download=True, train=False, )
    return mnist_train, mnist_test


# Next we do the preprocessing
def preprocess_mnist_factory(
        name="preprocess_mnist",
):
    """
    Args:
        name (str): The name of the new op.
        ins (Dict[str, In]): Any Ins for the new op. Default: None.

    Returns:
        function: The new op.
    """

    @op(name=name, )
    def preprocess_mnist(context, dataset: Dataset) -> Dataset:
        dataset.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

        context.log.info(dataset)

        return dataset

    return preprocess_mnist


@op
def preprocess_mnist(mnist: Dataset) -> Dataset:
    mnist.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    return mnist


@op(out={"train_dataset": Out(), "val_dataset": Out(), })
def split_train_val(context, dataset: Dataset) -> tuple[Dataset, Dataset]:
    indices = [i for i in range(len(dataset))]
    train_indices, test_indices = train_test_split(indices, test_size=0.25, )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    context.log.info(len(train_dataset))
    context.log.info(len(test_dataset))
    context.log.info(test_dataset[0])

    return train_dataset, test_dataset


@graph(out={"train_dataset": GraphOut(), "val_dataset": GraphOut(), "test_dataset": GraphOut(), })
def load_and_preprocess_mnist() -> tuple[Dataset, Dataset, Dataset]:
    '''
    Main entrypoint for loading and preprocessing mnist. It creates a train, val and test dataset.
    :return: train_dataset, val_dataset, test_dataset.
    '''
    train_dataset, test_dataset = mnist()
    preprocess_train = preprocess_mnist_factory("preprocess_train")
    preprocess_test = preprocess_mnist_factory("preprocess_test")
    train_dataset = preprocess_train(train_dataset)
    test_dataset = preprocess_test(test_dataset)

    train_dataset, val_dataset = split_train_val(train_dataset)
    return train_dataset, val_dataset, test_dataset
