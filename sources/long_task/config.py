import configparser
import os

DEFAULT_PROVIDER_FAMILY = 'deepseek'
DEFAULT_MAX_PATENTS = 20
DEFAULT_VISION_PROVIDER = 'minimax'
DEFAULT_VISION_MODEL = 'MiniMax-M3'


def get_long_task_config(config_path: str = 'config.ini') -> dict:
    """Read [LONG_TASK] section from config file.

    Returns:
        dict with keys:
            provider_family (str)  — 'deepseek' or 'minimax'
            max_patents (int)
            vision_enabled (bool)
            vision_provider (str)  — provider name for vision LLM
            vision_model (str)     — model name for vision LLM
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    provider_family = DEFAULT_PROVIDER_FAMILY
    max_patents = DEFAULT_MAX_PATENTS
    vision_provider = DEFAULT_VISION_PROVIDER
    vision_model = DEFAULT_VISION_MODEL

    if cfg.has_section('LONG_TASK'):
        provider_family = cfg.get('LONG_TASK', 'provider_family',
                                  fallback=DEFAULT_PROVIDER_FAMILY)
        max_patents = cfg.getint('LONG_TASK', 'max_patents',
                                 fallback=DEFAULT_MAX_PATENTS)
        vision_provider = cfg.get('LONG_TASK', 'vision_provider',
                                  fallback=DEFAULT_VISION_PROVIDER)
        vision_model = cfg.get('LONG_TASK', 'vision_model',
                               fallback=DEFAULT_VISION_MODEL)

    vision_enabled = cfg.getboolean('LONG_TASK', 'vision_enabled',
                                     fallback=True)

    return {
        'provider_family': provider_family,
        'max_patents': max_patents,
        'vision_enabled': vision_enabled,
        'vision_provider': vision_provider,
        'vision_model': vision_model,
    }
