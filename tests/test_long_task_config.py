import os
import tempfile
import pytest
from sources.long_task.config import get_long_task_config


def test_reads_provider_family_and_max_patents():
    """Should read provider_family and max_patents from [LONG_TASK] section."""
    config_content = """[LONG_TASK]
provider_family = deepseek
max_patents = 20
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = get_long_task_config(config_path)
        assert config['provider_family'] == 'deepseek'
        assert config['max_patents'] == 20
    finally:
        os.unlink(config_path)


def test_defaults_when_section_missing():
    """Should return defaults when [LONG_TASK] section is absent."""
    config_content = """[MAIN]
provider_name = openai
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = get_long_task_config(config_path)
        assert config['provider_family'] in ('deepseek', 'minimax')
        assert isinstance(config['max_patents'], int)
        assert config['max_patents'] > 0
    finally:
        os.unlink(config_path)


def test_max_patents_is_integer():
    """max_patents should always be returned as an int."""
    config_content = """[LONG_TASK]
max_patents = 15
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = get_long_task_config(config_path)
        assert isinstance(config['max_patents'], int)
        assert config['max_patents'] == 15
    finally:
        os.unlink(config_path)
