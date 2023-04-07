from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Dict, Tuple, Set, Iterable, Union

from matplotlib import pyplot as plt
import numpy as np
import math
import PIL.Image
import cv2
import torch
import torch.nn.functional as F

from .evaluate import compute_ioa
from .utils import auto_autocast

__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'PixelHeatMap', 'ParsedHeatMap', 'SyntacticHeatMapPair']


def plot_overlay_heat_map(im, heat_map, figsize: Tuple[int, int] =(10,10)):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor) -> None

    with auto_autocast(dtype=torch.float32):
        plt.figure(figsize=figsize)
        plt.axis('off')
        im = np.array(im)
        plt.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')

        im = torch.from_numpy(im).float() / 255
        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
        plt.imshow(im)


class PixelHeatMap:
    def __init__(self, heatmap: torch.Tensor):
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def plot_overlay(self, image, out_file=None, color_normalize=True, ax=None, **expand_kwargs):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        plot_overlay_heat_map(
            image,
            self.expand_as(image, **expand_kwargs),
            out_file=out_file,
            color_normalize=color_normalize,
            ax=ax
        )

    def expand_as(self, image, absolute=False, threshold=None, plot=False, **plot_kwargs):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor

        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')
        im = im[0,0]

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        im = im.cpu().detach().squeeze()

        if plot:
            self.plot_overlay(image, **plot_kwargs)

        return im

    def compute_ioa(self, other: 'PixelHeatMap'):
        return compute_ioa(self.heatmap, other.heatmap)


@dataclass
class SyntacticHeatMapPair:
    head_heat_map: PixelHeatMap
    dep_heat_map: PixelHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class ParsedHeatMap:
    word_heat_map: PixelHeatMap

class GlobalHeatMap:
    def __init__(self, heat_maps: torch.Tensor, latent_hw: int):
        self.heat_maps = heat_maps
        self.latent_h = self.latent_w = int(math.sqrt(latent_hw))

    def compute_pixel_heat_map(self, latent_pixels: Union[List[int], int] = None, influx: bool = False) -> PixelHeatMap:
        """
        Given a list of pixels or pixel id it returns the heatmap for that pixel or mean of all the heatmaps corresponding
        to those pixels.
        The pixel ids should adhere to row-major latent image representation i.e.
        0 1 ... 63
        ..........
        4032...4095
        for SDV2
        influx: Calculates the heatmap of all the pixels except the pixel_ids passed
        """
        if isinstance(latent_pixels, list):
            if not influx:
                merge_idxs = latent_pixels
            else: # If influx is true we we calculate the heatmap mean of all the pixels except the one passed
                merge_idxs = [p_id for p_id in range(self.latent_h * self.latent_w) if p_id not in latent_pixels]
            return PixelHeatMap(self.heat_maps[merge_idxs].mean(0))
        else:
            if not influx: 
                merge_idx = [latent_pixels]
            else: # If influx is true we we calculate the heatmap mean of all the pixels except the one passed
                merge_idx = [p_id for p_id in range(self.latent_h * self.latent_w) if p_id != latent_pixels]
            return PixelHeatMap(self.heat_maps[merge_idx].mean(0))

    def compute_bbox_heat_map(self, x1: int, y1: int, x2: int, y2: int, influx: bool = False) -> PixelHeatMap:
        """
        Given the top-left coordinates (x1,y1) and bottom-right coordinates (x2,y2) it returns the heatmap for the 
        mean of all the pixels lying inside this bbox.
        These coordinates should be for the latent image
        """
        if x2 < x1 or y2 < y1:
            raise Exception('Enter valid bounding box! (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner.')
        pix_ids = [x for y in range(y1, y2+1) for x in range((self.latent_w * y) + x1, (self.latent_w * y) + x2 + 1) if x < (self.latent_h * self.latent_w)]
        if influx: # If influx is true we we calculate the heatmap mean of all the pixels except the one passed
            pix_ids = [p_id for p_id in range(self.latent_h * self.latent_w) if p_id not in pix_ids]
        return PixelHeatMap(self.heat_maps[pix_ids].mean(0))

    def compute_contour_heat_map(self, pts: List[List[int]], image_h: int, image_w: int, influx: bool = False) -> PixelHeatMap:
        """
        pts should be be a list of [x,y] coordinates of the contour
        image_h and image_w is the image height and width respectively of the original image from which contour is taken
        returns the heatmap for the mean of the pixels lying inside this contour
        """
        if image_h != image_w:
            raise Exception('Non-Square images not supported yet! `image_h` should be equal to `image_w')

        pts = np.array(np.array(pts) * self.latent_h / image_h, np.int32)
        pts = pts.reshape((-1,1,2))
        inner_pixs = list()
        for i in range(self.latent_h):
            for j in range(self.latent_w):
                dist = cv2.pointPolygonTest(pts, (i, j), False)
                if dist == 1.0:
                    inner_pixs.append((j*self.latent_w) + i)
        if influx: # If influx is true we we calculate the heatmap mean of all the pixels except the one passed
            inner_pixs = [p_id for p_id in range(self.latent_h * self.latent_w) if p_id not in inner_pixs]

        return PixelHeatMap(self.heat_maps[inner_pixs].mean(0))

RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head


class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
