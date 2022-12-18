from dagster import op, Out
from optuna import Study
from torch.utils.data import Dataset


from models.feed_forward_network import LightningDnn
from models.hyperparameter_search import HyperparameterSearch



@op(out={"model": Out(), "study": Out(), })
def hyperparameter_search(train_dataset: Dataset, val_dataset: Dataset) -> tuple[LightningDnn, Study]:
    '''
    Perform hyperparameter search
    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :return: best model and best model config
    '''

    hyperparameter_search = HyperparameterSearch(train_dataset, val_dataset)
    study, model = hyperparameter_search.search()

    return model, study
