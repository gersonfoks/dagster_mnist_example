from dagster import repository, job

from pipelines.hyperparameter_search import hyperparameter_search
from pipelines.model_creation import load_config, create_model_from_config, save_model, load_model, create_model
from pipelines.preprocess import preprocess_mnist, mnist, split_train_val, load_and_preprocess_mnist
from pipelines.training import train_model, test_model


@job
def main_pipeline():
    '''
        The main pipeline for training an ML model
    '''
    train_dataset, val_dataset, test_dataset = load_and_preprocess_mnist()

    model, model_config = create_model()

    model = train_model(model, train_dataset, val_dataset)

    test_model(model, test_dataset)

    save_model(model, model_config)


@job
def hyperparametersearch_pipeline():
    """
    Pipeline for hyperparameter search
    """

    train_dataset, val_dataset, test_dataset = load_and_preprocess_mnist()

    best_model, best_model_config = hyperparameter_search(train_dataset, val_dataset)

    test_model(best_model, test_dataset)


@repository
def my_repository():
    return [
        main_pipeline,
        hyperparametersearch_pipeline,
    ]
