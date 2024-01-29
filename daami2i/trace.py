from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Union
import math

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F

from .utils import cache_dir, auto_autocast
from .experiment import GenerationExperiment
from .heatmap import RawHeatMapCollection, GlobalHeatMap
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator


__all__ = ['trace', 'DiffusionHeatMapHooker', 'GlobalHeatMap']


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
            self,
            pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
            low_memory: bool = False,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: str = None,
            track_all: bool = False
    ):
        self.all_heat_maps = RawHeatMapCollection()
        h = (pipeline.unet.config.sample_size * pipeline.vae_scale_factor)
        self.latent_hw = 4096 if h == 512 or h == 768 or h == 1024 else 9216  # 64x64 or 96x96 depending on if it's 2.0-v or 2.0
        locate_middle = load_heads or save_heads
        self.locator = UNetCrossAttentionLocator(restrict={0} if low_memory else None, locate_middle_block=locate_middle)
        self.last_prompt: str = ''
        self.last_image: Image = None
        self.time_idx = 0
        self._gen_idx = 0
        self.track_all = track_all

        modules = [
            UNetCrossAttentionHooker(
                x,
                self,
                context_size=self.latent_hw,
                layer_idx=idx,
                latent_hw=self.latent_hw,
                load_heads=load_heads,
                save_heads=save_heads,
                data_dir=data_dir,
                track_all=track_all
            ) for idx, x in enumerate(self.locator.locate(pipeline.unet))
        ]

        modules.append(PipelineHooker(pipeline, self))

        if type(pipeline) == StableDiffusionXLPipeline:
            modules.append(ImageProcessorHooker(pipeline.image_processor, self))

        super().__init__(modules)
        self.pipe = pipeline

    def time_callback(self, *args, **kwargs):
        self.time_idx += 1

    @property
    def layer_names(self):
        return self.locator.layer_names

    def compute_global_heat_map(self, factors=None, head_idx=None, layer_idx=None, normalize=False):
        # type: (str, List[float], int, int, bool) -> GlobalHeatMap
        """
        Compute the global heat map for each latent pixel, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing latent pixel-level heat maps.
        """
        heat_maps = self.all_heat_maps

        if factors is None:
            factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)

        all_merges = []
        x = int(np.sqrt(self.latent_hw))

        with auto_autocast(dtype=torch.float32):
            for (factor, layer, head), heat_map in heat_maps:
                if factor in factors and (head_idx is None or head_idx == head) and (layer_idx is None or layer_idx == layer):
                    heat_map = heat_map.unsqueeze(1)
                    # The clamping fixes undershoot.
                    if len(factors) > 1:
                        all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))
                    else:
                        all_merges.append(heat_map)

            try:
                maps = torch.zeros_like(all_merges[0])
                # maps = torch.stack(all_merges, dim=0)
                for map in all_merges:
                    maps += map
            except RuntimeError:
                if head_idx is not None or layer_idx is not None:
                    raise RuntimeError('No heat maps found for the given parameters.')
                else:
                    raise RuntimeError('No heat maps found. Did you forget to call `with trace(...)` during generation?')

            # maps = maps.mean(0)[:, 0]
            maps = maps / len(all_merges)
            maps = maps[:, 0]

            if normalize:
                maps = maps / (maps.sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities

        return GlobalHeatMap(maps, maps.shape[0])
class ImageProcessorHooker(ObjectHooker[VaeImageProcessor]):
    def __init__(self, processor: VaeImageProcessor, parent_trace: 'trace'):
        super().__init__(processor)
        self.parent_trace = parent_trace

    def _hooked_postprocess(hk_self, _: VaeImageProcessor, *args, **kwargs):
        images = hk_self.monkey_super('postprocess', *args, **kwargs)
        hk_self.parent_trace.last_image = images[0]

        return images

    def _hook_impl(self):
        self.monkey_patch('postprocess', self._hooked_postprocess)


class PipelineHooker(ObjectHooker[StableDiffusionPipeline]):
    def __init__(self, pipeline: StableDiffusionPipeline, parent_trace: 'trace'):
        super().__init__(pipeline)
        self.heat_maps = parent_trace.all_heat_maps
        self.parent_trace = parent_trace

    def _hooked_run_safety_checker(hk_self, self: StableDiffusionPipeline, image, *args, **kwargs):
        image, has_nsfw = hk_self.monkey_super('run_safety_checker', image, *args, **kwargs)

        if self.image_processor:
            if torch.is_tensor(image):
                images = self.image_processor.postprocess(image, output_type='pil')
            else:
                images = self.image_processor.numpy_to_pil(image)
        else:
            images = self.numpy_to_pil(image)

        hk_self.parent_trace.last_image = images[len(images)-1]

        return image, has_nsfw

    def _hooked_check_inputs(hk_self, _: StableDiffusionPipeline, prompt: Union[str, List[str]], *args, **kwargs):
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError('Only single prompt generation is supported for heat map computation.')
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt

        hk_self.heat_maps.clear()
        hk_self.parent_trace.last_prompt = last_prompt

        return hk_self.monkey_super('check_inputs', prompt, *args, **kwargs)

    def _hook_impl(self):
        self.monkey_patch('run_safety_checker', self._hooked_run_safety_checker, strict=False)  # not present in SDXL
        self.monkey_patch('check_inputs', self._hooked_check_inputs)


class UNetCrossAttentionHooker(ObjectHooker[Attention]):
    def __init__(
            self,
            module: Attention,
            parent_trace: 'trace',
            context_size: int = 4096,
            layer_idx: int = 0,
            latent_hw: int = 9216,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: Union[str, Path] = None,
            track_all: bool = False
    ):
        super().__init__(module)
        self.heat_maps = parent_trace.all_heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

        self.load_heads = load_heads
        self.save_heads = save_heads
        self.trace = parent_trace

        # Whether to track all attentions (even if the attention heatmaps are of different size)
        self.track_all = track_all

        if data_dir is not None:
            data_dir = Path(data_dir)
        else:
            data_dir = cache_dir() / 'heads'

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, height * width)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): self attention slice/map between the image pixel and image pixel.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                # map_ = map_[map_.size(0) // 2:]  # Filter out unconditional -- Not sure why this is there
                maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (height * width, heads, height, width)
        return maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, height * width, height, width)

    # Deprecated: Was valid in older version of DAAM-I2I
    # def _hooked_sliced_attention(hk_self, self, query, key, value, sequence_length, dim):
    #     batch_size_attention = query.shape[0]
    #     hidden_states = torch.zeros(
    #         (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
    #     )
    #     slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
    #     for i in range(hidden_states.shape[0] // slice_size):
    #         start_idx = i * slice_size
    #         end_idx = (i + 1) * slice_size
    #         attn_slice = torch.baddbmm(
    #             torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
    #             query[start_idx:end_idx],
    #             key[start_idx:end_idx].transpose(-1, -2),
    #             beta=0,
    #             alpha=self.scale,
    #         )
    #         attn_slice = attn_slice.softmax(dim=-1)
    #         factor = int(math.sqrt(hk_self.latent_hw // attn_slice.shape[1]))

    #         if attn_slice.shape[-1] == hk_self.context_size:
    #             # shape: (batch_size, 64 // factor, 64 // factor, 77)
    #             maps = hk_self._unravel_attn(attn_slice)

    #             for head_idx, heatmap in enumerate(maps):
    #                 hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap)

    #         attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

    #         hidden_states[start_idx:end_idx] = attn_slice

    #     # reshape hidden_states
    #     hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
    #     return hidden_states

    def _save_attn(self, attn_slice: torch.Tensor):
        torch.save(attn_slice, self.data_dir / f'{self.trace._gen_idx}.pt')

    def _load_attn(self) -> torch.Tensor:
        return torch.load(self.data_dir / f'{self.trace._gen_idx}.pt')

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states) 

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query) # shape = (batch_size * num_heads, latent_image_seq_len, 64) # For SDV2
        key = attn.head_to_batch_dim(key) # shape = (batch_size * num_heads, latent_image_seq_len, 64) # For SDV2
        value = attn.head_to_batch_dim(value) # shape = (batch_size * num_heads, latent_image_seq_len, 64) # For SDV2
        
        # shape: (batch_size * num_heads, latent_image_seq_len, latent_image_seq_len) # For SDV2: (batch_size * 10, 4096, 4096)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # DAAM save heads
        if self.save_heads:
            self._save_attn(attention_probs)
        elif self.load_heads:
            attention_probs = self._load_attn()

        # compute shape factor
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1]))
        self.trace._gen_idx += 1

        if not self.track_all:
            # skip if too large
            if attention_probs.shape[-1] == self.context_size and factor != 8:
                # shape: (batch_size, 64 // factor, 64 // factor, 77) --> Valid for DAAM, not for DAAM-I2I
                maps = self._unravel_attn(attention_probs) # shape: (heads, batch_size, height * width, height, width)

                for head_idx, heatmap in enumerate(maps):
                    self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)
        else:
            maps = self._unravel_attn(attention_probs) # shape: (heads, batch_size, height * width, height, width)

            for head_idx, heatmap in enumerate(maps):
                self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    # Deprecated: Was valid in older version of DAAM-I2I
    # def _hooked_attention(hk_self, self, query, key, value):
    #     """
    #     Monkey-patched version of :py:func:`.CrossAttention._attention` to capture attentions and aggregate them.
    #     Args:
    #         hk_self (`UNetCrossAttentionHooker`): pointer to the hook itself.
    #         self (`CrossAttention`): pointer to the module.
    #         query (`torch.Tensor`): the query tensor. shape: (batch_size * heads, height * width, image_embedding_size)
    #         key (`torch.Tensor`): the key tensor. shape: (batch_size * heads, height * width, image_embedding_size)
    #         value (`torch.Tensor`): the value tensor. shape: (batch_size * heads, height * width, image_embedding_size)
    #     """
    #     # query.shape = (batch_size * num_heads, latent_image_seq_len, 64) # For SDV2
    #     # key.shape = (batch_size * num_heads, latent_image_seq_len, 64) # For SDV2
    #     # value.shape = (batch_size * num_heads, latent_image_seq_len, 64) # For SDV2

    #     attention_scores = torch.baddbmm(
    #         torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
    #         query,
    #         key.transpose(-1, -2),
    #         beta=0,
    #         alpha=self.scale,
    #     ) # shape: (batch_size * num_heads, latent_image_seq_len, latent_image_seq_len) # For SDV2: (batch_size * 10, 4096, 4096)

    #     attn_slice = attention_scores.softmax(dim=-1) # shape: (batch_size * 10, 4096, 4096)

    #     if hk_self.save_heads: # save the attention slices locally if wanting to inspect
    #         hk_self._save_attn(attn_slice)
    #     elif hk_self.load_heads:
    #         attn_slice = hk_self._load_attn()

    #     # Factor can be thought of as the levels of the UNet
    #     factor = int(math.sqrt(hk_self.latent_hw // attn_slice.shape[1]))
    #     hk_self.trace._gen_idx += 1

    #     if not hk_self.track_all:
    #         if attn_slice.shape[-1] == hk_self.context_size and factor != 8:
    #             # shape: (batch_size, 64 // factor, 64 // factor, 77)
    #             maps = hk_self._unravel_attn(attn_slice) # shape: (heads, batch_size, height * width, height, width)

    #             for head_idx, heatmap in enumerate(maps):
    #                 hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap) # heatmap shape: (batch_size, height * width, height, width)
    #     else:
    #         # shape: (batch_size, 64 // factor, 64 // factor, 77)
    #         maps = hk_self._unravel_attn(attn_slice) # shape: (heads, batch_size, height * width, height, width)

    #         for head_idx, heatmap in enumerate(maps):
    #             hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap) # heatmap shape: (batch_size, height * width, height, width)

    #     # compute attention output
    #     hidden_states = torch.bmm(attn_slice, value)

    #     # torch.save(hidden_states, f'{factor}-{hk_self.trace._gen_idx}.pt')

    #     # reshape hidden_states
    #     hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
    #     return hidden_states

    def _hook_impl(self):
        self.original_processor = self.module.processor
        self.module.set_processor(self)

    def _unhook_impl(self):
        self.module.set_processor(self.original_processor)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
