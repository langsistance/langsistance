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
DEFAULT_PROSECUTION_INCLUDE_PRIORITY_2 = False


# ── Patent family analysis config ───────────────────────────────────────────────

DEFAULT_EPO_CONSUMER_KEY = ''
DEFAULT_EPO_CONSUMER_SECRET = ''
DEFAULT_FAMILY_MAX_JURISDICTIONS = 1  # start with US only


def get_family_config(config_path: str = 'config.ini') -> dict:
    """Read [FAMILY] section from config file.

    Returns:
        dict with keys:
            epo_consumer_key (str)
            epo_consumer_secret (str)
            max_jurisdictions (int)
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    epo_consumer_key = DEFAULT_EPO_CONSUMER_KEY
    epo_consumer_secret = DEFAULT_EPO_CONSUMER_SECRET
    max_jurisdictions = DEFAULT_FAMILY_MAX_JURISDICTIONS

    if cfg.has_section('FAMILY'):
        epo_consumer_key = cfg.get('FAMILY', 'epo_consumer_key',
                                   fallback=DEFAULT_EPO_CONSUMER_KEY)
        epo_consumer_secret = cfg.get('FAMILY', 'epo_consumer_secret',
                                      fallback=DEFAULT_EPO_CONSUMER_SECRET)
        max_jurisdictions = cfg.getint('FAMILY', 'max_jurisdictions',
                                        fallback=DEFAULT_FAMILY_MAX_JURISDICTIONS)

    return {
        'epo_consumer_key': epo_consumer_key,
        'epo_consumer_secret': epo_consumer_secret,
        'max_jurisdictions': max_jurisdictions,
    }


# ── SIPOP (China patent open data platform) config ──────────────────────────

DEFAULT_SIPOP_APP_KEY = ''
DEFAULT_SIPOP_APP_SECRET = ''


def get_sipop_config(config_path: str = 'config.ini') -> dict:
    """Read [SIPOP] section from config file, with env var overrides.

    Returns:
        dict with keys: app_key (str), app_secret (str)
    """
    import os as _os
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    app_key = _os.getenv('SIPOP_APP_KEY', '')
    app_secret = _os.getenv('SIPOP_APP_SECRET', '')

    if not app_key and cfg.has_section('SIPOP'):
        app_key = cfg.get('SIPOP', 'app_key', fallback=DEFAULT_SIPOP_APP_KEY)
    if not app_secret and cfg.has_section('SIPOP'):
        app_secret = cfg.get('SIPOP', 'app_secret', fallback=DEFAULT_SIPOP_APP_SECRET)

    return {
        'app_key': app_key.strip(),
        'app_secret': app_secret.strip(),
    }


DEFAULT_JPO_USERNAME = ''
DEFAULT_JPO_PASSWORD = ''


def get_jpo_config(config_path: str = 'config.ini') -> dict:
    """Read [JPO] section from config file, with env var overrides.

    Returns:
        dict with keys: username (str), password (str)
    """
    import os as _os
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    username = _os.getenv('JPO_USERNAME', '')
    password = _os.getenv('JPO_PASSWORD', '')

    if not username and cfg.has_section('JPO'):
        username = cfg.get('JPO', 'username', fallback=DEFAULT_JPO_USERNAME)
    if not password and cfg.has_section('JPO'):
        password = cfg.get('JPO', 'password', fallback=DEFAULT_JPO_PASSWORD)

    return {
        'username': username.strip(),
        'password': password.strip(),
    }


def get_prosecution_config(config_path: str = 'config.ini') -> dict:
    """Read [PROSECUTION] section from config file.

    Returns:
        dict with keys:
            max_pages_per_doc (int)      — max pages to download per document (0 = unlimited)
            include_priority_2 (bool)     — whether to include IDS, Interview Summary etc.
            streaming_provider (str|None) — override LLM provider for streaming report output
            streaming_model (str|None)    — override LLM model for streaming report output
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    max_pages_per_doc = DEFAULT_PROSECUTION_MAX_PAGES_PER_DOC
    include_priority_2 = DEFAULT_PROSECUTION_INCLUDE_PRIORITY_2
    streaming_provider = None
    streaming_model = None

    if cfg.has_section('PROSECUTION'):
        max_pages_per_doc = cfg.getint('PROSECUTION', 'max_pages_per_doc',
                                        fallback=DEFAULT_PROSECUTION_MAX_PAGES_PER_DOC)
        include_priority_2 = cfg.getboolean('PROSECUTION', 'include_priority_2',
                                             fallback=DEFAULT_PROSECUTION_INCLUDE_PRIORITY_2)
        streaming_provider = cfg.get('PROSECUTION', 'streaming_provider',
                                      fallback=None) or None
        streaming_model = cfg.get('PROSECUTION', 'streaming_model',
                                   fallback=None) or None

    return {
        'max_pages_per_doc': max_pages_per_doc,
        'include_priority_2': include_priority_2,
        'streaming_provider': streaming_provider,
        'streaming_model': streaming_model,
    }
