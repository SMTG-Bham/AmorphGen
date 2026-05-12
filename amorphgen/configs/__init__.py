"""
amorphgen.configs
----------------
Default configuration for all 7 pipeline stages.

    from amorphgen.configs import DEFAULT_CONFIG
    from amorphgen.configs import load_yaml_config
"""

from .default_config import DEFAULT_CONFIG
from .yaml_config import load_yaml_config

__all__ = ["DEFAULT_CONFIG", "load_yaml_config"]
