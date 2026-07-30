"""Config schema, secret references, and the loader (SPEC §6)."""

from imsg.config.loader import (
    assert_secrets_resolvable,
    default_config_path,
    load_config,
    load_config_dict,
)
from imsg.config.schema import Config
from imsg.config.secrets import SecretRef

__all__ = [
    "Config",
    "SecretRef",
    "assert_secrets_resolvable",
    "default_config_path",
    "load_config",
    "load_config_dict",
]
