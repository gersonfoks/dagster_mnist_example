# Introduction

This projects gives an example on how to use Dagster to build a simple training pipeline.

## Overview

The pipeline is composed of the following steps:

- Preprocessing the data
- Model creation
- Training the model
- Evaluation of the model

With this setup we are also able to reuse the steps in other pipelines. Such as a hyperparameter tuning pipeline.

# Setup

# run the pipeline

To open the pipeline in dagit, you can use the following command:

```bash

dagit -f main_pipeline.py

```

There are two pipelines defined in this file:
- `main_pipeline` is the main pipeline
- `hyperparametersearch_pipeline` is used for a hyperparameter search




# TODO

- [x] Add training step
- [x] Add evaluation step
- [x] Add a hyperparameter tuning pipeline
- [x] Add a saving model step 


