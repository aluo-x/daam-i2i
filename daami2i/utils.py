from functools import lru_cache
from pathlib import Path
import os
import sys
import random
from typing import TypeVar

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


__all__ = ['set_seed', 'plot_mask_heat_map', 'cache_dir', 'auto_device', 'auto_autocast']


T = TypeVar('T')


def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj


def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs['enabled'] = False

    return torch.cuda.amp.autocast(*args, **kwargs)


def plot_mask_heat_map(im: PIL.Image.Image, heat_map: torch.Tensor, threshold: float = 0.4):
    im = torch.from_numpy(np.array(im)).float() / 255
    mask = (heat_map.squeeze() > threshold).float()
    im = im * mask.unsqueeze(-1)
    plt.imshow(im)


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen


def cache_dir() -> Path:
    # *nix
    if os.name == 'posix' and sys.platform != 'darwin':
        xdg = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        return Path(xdg, 'daam')
    elif sys.platform == 'darwin':
        # Mac OS
        return Path(os.path.expanduser('~'), 'Library/Caches/daam')
    else:
        # Windows
        local = os.environ.get('LOCALAPPDATA', None) \
                or os.path.expanduser('~\\AppData\\Local')
        return Path(local, 'daam')