# MNIST Classifier with PyTorch Lightning and Hydra
The purpose of this project is to learn about the
[PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and
[Hydra](https://hydra.cc/docs/intro/) packages. They both offer great utility
for simplifying the process of creating and using pytorch for a project.

Lightning offers the `Trainer` module, which automates a lot of the typical
pytorch code, so we are free to focus more on the model (architecture and low
level details of training, testing, and logging) and data.

Hydra is a configuration management tool that allows you to specify all the
settings and hyperparameters in separate configuration files.  Splitting them
from the code means we do not need to change the code at all in order to change
how it runs, or even what data and/or model to use. In this example, however,
there is only a single model and single datatype

## Running the code on Colab
The other purpose of this project is to test cloning a repo in a Google Colab
notebook, and using that GPU environment to run the training, testing, and
hyperparameter sweeps.

## References
This project was heavily influenced by these project templates:
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [Lightning-AI/deep-learning-project-template](https://github.com/Lightning-AI/deep-learning-project-template)


## TODO
- [x] add more to the configs
    - [x] settings for output directories
    - [x] basic hparams search
    - [x] use the hydra instantiation
- [ ] ~~create a conda environment~~
- [x] create a train (and eval?) script for actually running things
- [x] ~~test locally~~, then run hparam search with GPU on google colab
- [ ] update logging so that lightning logging is adding to the same directory
  as hydra

