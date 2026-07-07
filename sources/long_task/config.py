import configparser
import os

DEFAULT_PROVIDER_FAMILY = 'deepseek'
DEFAULT_MAX_PATENTS = 20
DEFAULT_MAX_PATENTS_CNIPA = 10
DEFAULT_MAX_PATENTS_USPTO = 50
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
    max_patents_cnipa = DEFAULT_MAX_PATENTS_CNIPA
    max_patents_uspto = DEFAULT_MAX_PATENTS_USPTO
    vision_provider = DEFAULT_VISION_PROVIDER
    vision_model = DEFAULT_VISION_MODEL

    if cfg.has_section('LONG_TASK'):
        provider_family = cfg.get('LONG_TASK', 'provider_family',
                                  fallback=DEFAULT_PROVIDER_FAMILY)
        max_patents = cfg.getint('LONG_TASK', 'max_patents',
                                 fallback=DEFAULT_MAX_PATENTS)
        max_patents_cnipa = cfg.getint('LONG_TASK', 'max_patents_cnipa',
                                        fallback=DEFAULT_MAX_PATENTS_CNIPA)
        max_patents_uspto = cfg.getint('LONG_TASK', 'max_patents_uspto',
                                        fallback=DEFAULT_MAX_PATENTS_USPTO)
        vision_provider = cfg.get('LONG_TASK', 'vision_provider',
                                  fallback=DEFAULT_VISION_PROVIDER)
        vision_model = cfg.get('LONG_TASK', 'vision_model',
                               fallback=DEFAULT_VISION_MODEL)

    vision_enabled = cfg.getboolean('LONG_TASK', 'vision_enabled',
                                     fallback=True)

    return {
        'provider_family': provider_family,
        'max_patents': max_patents,
        'max_patents_cnipa': max_patents_cnipa,
        'max_patents_uspto': max_patents_uspto,
        'vision_enabled': vision_enabled,
        'vision_provider': vision_provider,
        'vision_model': vision_model,
    }


# ── Prosecution analysis config ───────────────────────────────────────────────

DEFAULT_PROSECUTION_MAX_PAGES_PER_DOC = 100
DEFAULT_PROSECUTION_INCLUDE_PRIORITY_2 = True


def get_prosecution_config(config_path: str = 'config.ini') -> dict:
    """Read [PROSECUTION] section from config file.

    Returns:
        dict with keys:
            max_pages_per_doc (int)  — max pages to download per document (0 = unlimited)
            include_priority_2 (bool) — whether to include IDS, Interview Summary etc.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    max_pages_per_doc = DEFAULT_PROSECUTION_MAX_PAGES_PER_DOC
    include_priority_2 = DEFAULT_PROSECUTION_INCLUDE_PRIORITY_2

    if cfg.has_section('PROSECUTION'):
        max_pages_per_doc = cfg.getint('PROSECUTION', 'max_pages_per_doc',
                                        fallback=DEFAULT_PROSECUTION_MAX_PAGES_PER_DOC)
        include_priority_2 = cfg.getboolean('PROSECUTION', 'include_priority_2',
                                             fallback=DEFAULT_PROSECUTION_INCLUDE_PRIORITY_2)

    return {
        'max_pages_per_doc': max_pages_per_doc,
        'include_priority_2': include_priority_2,
    }
