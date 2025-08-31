import pytest
import yaml
import tempfile
import os
from utils.config_loader import ConfigParser


@pytest.fixture
def valid_yaml_file():
    data = {
        "paths": {"train": "data/train", "val": "data/val"},
        "classes": ["cat", "dog"],
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as tmp:
        yaml.dump(data, tmp)
    yield path
    os.remove(path)


@pytest.fixture
def invalid_yaml_file():
    data = {"paths": {"train": "data/train"}}
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as tmp:
        yaml.dump(data, tmp)
    yield path
    os.remove(path)


def test_valid_config(valid_yaml_file):
    parser = ConfigParser(config_path=valid_yaml_file)
    config = parser.get_data()
    assert "paths" in config
    assert "classes" in config
    assert config["classes"] == ["cat", "dog"]


def test_missing_required_key(invalid_yaml_file):
    parser = ConfigParser(config_path=invalid_yaml_file)
    with pytest.raises(ValueError, match="Missing required config key: classes"):
        parser.get_data()


def test_file_not_found():
    parser = ConfigParser(config_path="non_existent.yaml")
    with pytest.raises(FileNotFoundError):
        parser.get_data()
