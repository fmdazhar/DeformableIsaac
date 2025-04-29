


import hydra
from omegaconf import DictConfig, OmegaConf

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
if not OmegaConf.has_resolver("eq"):
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
if not OmegaConf.has_resolver("contains"):
    OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
if not OmegaConf.has_resolver("if"):
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
if not OmegaConf.has_resolver("resolve_default"):
    OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)
