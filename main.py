import logging

from omegaconf import DictConfig, OmegaConf
import hydra

from core.config_factory import (
    create_config_factory,
    instantiate_configs,
    omegaconf_to_dict,
)
from core.simulation_manager import SimulationManager

logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="config", version_base=None)
def run(cfg: DictConfig) -> None:
    cfg_dict = omegaconf_to_dict(cfg)
    factory = create_config_factory()
    cfg_dict = instantiate_configs(cfg_dict, factory)

    logger.info("Configuration loaded: environment=%s", cfg_dict.get("name", "unknown"))

    sim = SimulationManager(cfg_dict)
    sim.setup()
    sim.run()
    sim.shutdown()


if __name__ == "__main__":
    run()
