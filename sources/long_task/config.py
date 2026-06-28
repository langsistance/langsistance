import configparser
import os

DEFAULT_PROVIDER_FAMILY = 'deepseek'
DEFAULT_MAX_PATENTS = 20


def get_long_task_config(config_path: str = 'config.ini') -> dict:
    """Read [LONG_TASK] section from config file.

    Returns:
        dict with keys: provider_family (str), max_patents (int)
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    provider_family = DEFAULT_PROVIDER_FAMILY
    max_patents = DEFAULT_MAX_PATENTS

    if cfg.has_section('LONG_TASK'):
        provider_family = cfg.get('LONG_TASK', 'provider_family',
                                  fallback=DEFAULT_PROVIDER_FAMILY)
        max_patents = cfg.getint('LONG_TASK', 'max_patents',
                                 fallback=DEFAULT_MAX_PATENTS)

    vision_enabled = cfg.getboolean('LONG_TASK', 'vision_enabled',
                                     fallback=True)

    return {
        'provider_family': provider_family,
        'max_patents': max_patents,
        'vision_enabled': vision_enabled,
    }
