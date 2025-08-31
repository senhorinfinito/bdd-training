import yaml
from utils.custom_logger import logger, one_line_symbol

__module_name__ = "[Config Parser]"


class ConfigParser:
    REQUIRED_KEYS = ["paths", "classes"]

    def __init__(self, config_path="./cfg/default.yaml"):
        self.config_path = config_path

    def __load_yaml(self):
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    def __validate_config(self, config):
        for key in self.REQUIRED_KEYS:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        return True

    def get_data(self):
        config = self.__load_yaml()
        self.__validate_config(config)
        logger.info(
            f"{__module_name__} : Configuration loaded successfully :  {config}"
        )
        one_line_symbol()
        return config


if __name__ == "__main__":
    config = ConfigParser().get_data()
