from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Dict, Tuple, Set, Iterable, Union, Optional

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


def plot_overlay_heat_map(im, heat_map, figsize: Tuple[int, int] = (10,10)):
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

    def plot_overlay(self, image, figsize: Tuple[int, int] = (10,10)):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        plot_overlay_heat_map(image, self.expand_as(image), figsize)

    def expand_as(self, image):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor

        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')
        im = im[0,0]
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        im = im.cpu().detach().squeeze()
        
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

    def compute_contour_heat_map(self, 
      pts: Union[List[List[int]], List[int]], 
      image_h: int, image_w: int, influx: bool = False, 
      guide_heatmap: Optional[torch.tensor] = None) -> PixelHeatMap:
        """
        pts should be a represent a single polygon multi-piece polygon is not handled by this function, check out `segmentation_heat_map`
        pts should be be a list of [x,y] coordinates of the contour 
            or a list of [x1,y1,x2,y2,...,xn,yn] (i.e. same as [[x1,y1], [x2.y2], ...] just the inner lists are unravelled)
        image_h and image_w is the image height and width respectively of the original image from which contour is taken
        guide_heatmap is the same as described in `compute_guided_heat_map`, in this case only the pixels of the `guide_heatmap` will be considered which are 
        contained inside the contour
        returns the heatmap for the mean of the pixels lying inside this contour
        """
        if isinstance(pts[0], int):
            pts = [[pts[i], pts[i+1]] for i in range(0, len(pts), 2)]

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

        if guide_heatmap is None:
            return PixelHeatMap(self.heat_maps[inner_pixs].mean(0))
        else:
            # Finding out the inner_pixs' each pixel's weight as obtained from `guide_heatmap`
            pix_weights = torch.tensor([guide_heatmap[pix_id // self.latent_w, pix_id % self.latent_w] for pix_id in inner_pixs])[:,None,None]
            # return the weighted heatmap
            return PixelHeatMap((self.heat_maps[inner_pixs] * pix_weights[:, None, None] / pix_weights.sum().item()).mean(0))

    def compute_segmentation_heat_map(self, 
      segments: Union[List[List[List[int]]], List[List[int]]], 
      image_h: int, image_w: int, 
      segment_weights: Optional[torch.tensor] = None, guide_heatmap: Optional[torch.tensor] = None) -> PixelHeatMap:
        """
        Pass in the list of contours like this [[x1,y1,x2,y2,....], [p1,q1,p2,q2,...], ..] or [[[x1,y1],[x2,y2],....], [[p1,q1],[p2,q2],...], ..]
        This finds the mean heatmap for all the pixel heatmaps for the pixels lying inside each of these contours together.
        segments: list of contours in the format explained above
        image_h: the height of the image according to which the `segments` is provided
        image_w: the width of the image according to which the `segments` is provided
        segment_weights: 1D tensor of the weight to be given to each segment in `segments` must be of the same length as the number of segment in `segments`
        """
        segment_heatmaps = list()
        for segment in segments:
            # Compute heatmap for inner pixels for contour boundary specified
            segment_heatmap = self.compute_contour_heat_map(segment, image_h, image_w, guide_heatmap=guide_heatmap)

            segment_heatmaps.append(segment_heatmap.heatmap)

        if segment_weights is None:
            return PixelHeatMap(torch.stack(segment_heatmaps).mean(0))
        else:
            return PixelHeatMap((torch.stack(segment_heatmaps) * segment_weights[:, None, None] / segment_weights.sum().item()).mean(0))


    def compute_guided_heat_map(self, guide_heatmap: torch.tensor) -> PixelHeatMap:
        """
        For each pixel in the latent image we have one heatmap. Now, with a guiding heatmap
        we can merge all these pixel heatmaps with a weighted average according to the weights 
        given to each pixel in the guiding heatmap. 

        guide_heatmap: A guiding heatmap of the dimension of the latent image. It should be a 2D torch.tensor
        """

        # To store weighted average of all the heatmaps with weights given in the `guide_heatmap`
        heatmap = torch.zeros((self.latent_h, self.latent_w)).to('cuda' if self.heatmap.get_device() == 0 else 'cpu')

        for i in range(self.latent_h):
            for j in range(self.latent_w):
                heatmap += self.compute_pixel_heat_map(self.latent_w * i + j).heatmap * guide_heatmap[i][j].item()

        heatmap /= guide_heatmap.sum().item()

        return PixelHeatMap(heatmap)

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
