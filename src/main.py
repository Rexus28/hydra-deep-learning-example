import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

hydra_log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config",
            config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """main function for running the deep learning models via the hydra configs

    Run `$ python src/main.py --help` to get more information on the available
    configuration options."""

    # -----------------------------------------
    # log the configuration in the hydra logger
    # -----------------------------------------
    hydra_log.info(OmegaConf.to_yaml(cfg))

    # -----------------------------------------
    # create the datamodule
    # -----------------------------------------
    hydra_log.info(f"Creating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # -----------------------------------------
    # create the model
    # -----------------------------------------
    hydra_log.info(f"Creating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # -----------------------------------------
    # create the trainer
    # -----------------------------------------
    hydra_log.info(f"Creating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # -----------------------------------------
    # start the training
    # -----------------------------------------
    hydra_log.info("Starting the training!")
    trainer.fit(model=model, datamodule=datamodule)

    # -----------------------------------------
    # start the testing
    # -----------------------------------------
    hydra_log.info("Starting the testing!")
    results = trainer.test(model=model, datamodule=datamodule)
    hydra_log.info(results)


if __name__ == "__main__":
    main()
