# Introduction

This projects gives an example on how to use Dagster to build a simple training and hyperparameter search pipeline.

It shows how you can create assets, jobs, ops, graphs and repositories. The main entrypoint is the 'main_pipeline.py'
file.

# Usage

To open the pipeline in dagit, you can use the following command:

```bash

dagit -f main_pipeline.py

```

There are two pipelines defined:

- `main_pipeline` is the main pipeline
- `hyperparametersearch_pipeline` is used for a hyperparameter search

# TODO

- [x] Add training step
- [x] Add evaluation step
- [x] Add a hyperparameter tuning pipeline
- [x] Add a saving model step 
- [ ] Add a saving hyperparameter search results step.



