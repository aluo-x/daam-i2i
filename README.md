# DAAM-Image2Image: Extension of DAAM for Image Self-Attention in Diffusion Models

The original DAAM was suitable for Token Heatmaps from Cross-Attention over Latent Image as shown below.
![example image](example.jpg)

But there for full utilization of the attention heatmaps, the Latent Image Self-Attention heatmap was also necessary. That is what this extention is for.

In [original DAAM paper](https://arxiv.org/abs/2210.04885), the author proposes diffusion attentive attribution maps (DAAM), a cross attention-based approach for interpreting Stable Diffusion for interpretability of token heatmap over latent images. Here, I use the same approach but extended for latent image self-attention heatmaps.

## Getting Started
First, install [PyTorch](https://pytorch.org) for your platform. You may check out the [Colab Tutorial](https://github.com/RishiDarkDevil/Text-Based-Object-Discovery/blob/main/Experiments/DAAM_Image_Attention_ver2.ipynb)

### Installation
The following steps are useful for setting up `daami2i` package in Colab Environment.

```
!git clone https://github.com/RishiDarkDevil/daam-i2i.git
%cd daam-i2i
!pip install -r requirements.txt
```

### Using DAAM as a Library

Import and use DAAM as follows:

```python
# Plotting
from matplotlib import pyplot as plt

# Data Handling
import numpy as np

# Image Processing
from PIL import Image

# Image Generation
from diffusers import StableDiffusionPipeline
import daami2i

DEVICE = 'cuda' # device

model = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-base')
model = model.to(DEVICE) # Set it to something else if needed, make sure DAAM supports that

prompt = 'Dinner table with chairs and flower pot'

# Image generation
with daami2i.trace(model) as trc:
  output_image = model(prompt).images
  global_heat_map = trc.compute_global_heat_map()
  
print(output_image[0]) # Output image
```
There are 3 types of visualizations available:
- Pixel-based here. The pixels are numbered in row-major order i.e.
  - 1     2 .. 64\
  4033 4034 .. 4096\
  Only latent image height and width is valid (i.e. 64 x 64 for Stable Diffusion v2 base) so the pixels that can be mentioned is a list from 1 ... 4096.
  ```python
  # Compute heatmap for latent pixel lists row-major
  pixel_heatmap = global_heat_map.compute_pixel_heat_map(list(range(1024))).expand_as(output_image[0]).numpy()

  # Casting heatmap from 0-1 floating range to 0-255 unsigned 8 bit integer
  heatmap = np.array(pixel_heatmap * 255, dtype = np.uint8)
  
  plt.imshow(heatmap)
  ```

- BBox based here. The bounding box upper left and bottom right corner needs to be specified. Again latent height and width are valid ranges.
  ```python
  # Compute heatmap for latent bbox pixels with corners specified
  pixel_heatmap = global_heat_map.compute_bbox_heat_map(0,10,25,64).expand_as(output_image[0]).numpy()
  
  # Casting heatmap from 0-1 floating range to 0-255 unsigned 8 bit integer
  heatmap = np.array(pixel_heatmap * 255, dtype = np.uint8)
  
  plt.imshow(heatmap)
  ```

- Contour based here. The image height and width can be different from the latent height and width. Enter contour and attention map will be generated for that contour containing pixels.
  ```python
  # Compute heatmap for inner pixels for contour boundary specified
  pixel_heatmap = global_heat_map.compute_contour_heat_map([[0,300], [256, 100], [512, 300], [512, 400], [0, 400], [0, 300]], 512, 512).expand_as(output_image[0]).numpy()

  # Casting heatmap from 0-1 floating range to 0-255 unsigned 8 bit integer
  heatmap = np.array(pixel_heatmap * 255, dtype = np.uint8)

  plt.imshow(heatmap)
  ```



## Citation

Original DAAM
```
@article{tang2022daam,
  title={What the {DAAM}: Interpreting Stable Diffusion Using Cross Attention},
  author={Tang, Raphael and Liu, Linqing and Pandey, Akshat and Jiang, Zhiying and Yang, Gefei and Kumar, Karun and Stenetorp, Pontus and Lin, Jimmy and Ture, Ferhan},
  journal={arXiv:2210.04885},
  year={2022}
}
```
