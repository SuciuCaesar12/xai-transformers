import math
import cv2
from .core import XaiDetrObjectDetectionOutput, XaiDetectionItem

from PIL import Image
from torchvision.utils import draw_bounding_boxes, make_grid, draw_segmentation_masks
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch.nn.functional as F
import torch
import numpy as np

import transformers as hf

from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union


ImageType: TypeAlias = Union[Image.Image, torch.Tensor]


def img_to_pt(image: ImageType) -> torch.Tensor:
    return pil_to_tensor(image) if isinstance(image, Image.Image) else image

def random_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    return [tuple(np.random.randint(0, 256, 3)) for _ in range(num_colors)]

def draw_detection(img_tensor, box, label, score, color):
    return draw_bounding_boxes(
        image=img_tensor,
        boxes=box.unsqueeze(0),
        labels=[f"{label} ({score:.2f})"],
        colors=[color],
        width=3
    )

def to_heatmap(map: torch.Tensor, color_map: int = cv2.COLORMAP_HOT) -> torch.Tensor:
    normalized_map = (map / map.max() * 255.).byte().numpy()
    heatmap = cv2.applyColorMap(normalized_map, color_map)
    return torch.tensor(heatmap[..., [2, 1, 0]]).permute(2, 0, 1)


class DetrVisualizer:
    
    def __init__(
        self,
        processor: hf.DetrImageProcessor,
        id2label: Dict[int, str]
    ):
        self.processor = processor
        self.id2label = id2label

    def _postprocess(
        self,
        image: torch.Tensor, 
        output: XaiDetrObjectDetectionOutput
    ) -> Dict[str, Any]:
        decoded_outputs = self.processor.post_process_object_detection(
            outputs=output.unsqueeze(),
            threshold=0,
            target_sizes=[image.shape[-2:]]
        )[0]
        output = output.squeeze()
        decoded_outputs['labels'] = list(map(self.id2label.get, decoded_outputs['labels'].tolist()))
        
        return decoded_outputs

    def draw_detections(
        self,
        image: ImageType,
        output: XaiDetrObjectDetectionOutput,
        to_pil: bool = False,
        together: bool = False,
        colors: Optional[List[Tuple[int, int, int]]] = None
    ):
        assert not output.is_batched(), "Batched outputs are not supported"
        
        img_tensor = img_to_pt(image)
        decoded_outputs = self._postprocess(img_tensor, output)
        
        boxes, labels, scores = decoded_outputs['boxes'], decoded_outputs['labels'], decoded_outputs['scores']
        colors = colors or random_colors(len(boxes))

        if not together:
            result = []
            for box, label, score, color in zip(boxes, labels, scores, colors):
                result.append(draw_detection(img_tensor, box, label, score, color))
            result = tuple(result)
            
            if to_pil:
                return tuple(to_pil_image(img) for img in result)
            return result
        else:
            result = draw_bounding_boxes(
                image=img_tensor,
                boxes=boxes,
                labels=[f'{l} ({s:.2f})' for l, s in zip(labels, scores)],
                colors=colors,
                width=3
            )
            
            if to_pil:
                return to_pil_image(result)
            return result
    
    def draw_relevance_maps(
        self, 
        output: XaiDetrObjectDetectionOutput, 
        to_pil: bool = False
    ):
        assert not output.is_batched(), "Batched outputs are not supported"
        heatmaps = tuple(to_heatmap(map) for map in output.relevance_maps)
        
        if to_pil:
            return tuple(to_pil_image(hm) for hm in heatmaps)
        return heatmaps

    def draw_detections_and_relevance_maps(
        self,
        image: Union[Image.Image, torch.Tensor],
        output: XaiDetrObjectDetectionOutput,
        to_pil: bool = False,
        grid: bool = False,
        colors: Optional[List[Tuple[int, int, int]]] = None
    ):
        assert not output.is_batched(), "Batched outputs are not supported"
        img_tensor = img_to_pt(image)
        img_size = img_tensor.shape[-2:]
        
        ds = self.draw_detections(img_tensor, output, colors)
        rel_maps = self.draw_relevance_maps(output)
        
        result = []
        for d, rm in zip(ds, rel_maps):
            resized_rm = F.interpolate(rm.unsqueeze(0), size=img_size, mode='bilinear').squeeze(0)
            result.append(make_grid([d, resized_rm], nrow=2, padding=20, pad_value=255.))

        if grid:
            nrow = math.ceil(math.sqrt(len(result)))
            result = make_grid(result, nrow=nrow, padding=20, pad_value=255.)
        
        if to_pil:
            result = to_pil_image(result) if grid else tuple(to_pil_image(img) for img in result)

        return result
    
    def generate_frames_for_token_evaluation_v2(
        self,
        image: Image.Image,
        d_t: XaiDetectionItem,
        d_p: List[XaiDetectionItem],
        masks: torch.Tensor,
        colors: List[Tuple[int, int, int]] = [(0, 0, 255), (0, 255, 0)],
        mask_color: Tuple[int, int, int] = (0, 0, 0),
        alpha: float = 0.7
    ):
        frames = []
        masks = F.interpolate(
            masks.unsqueeze(1), 
            size=image.size[::-1], 
            mode='nearest'
        )
        
        for d, mask in zip(d_p, list(masks)):
            frame = draw_segmentation_masks(
                image=img_to_pt(image),
                masks=~mask.bool(),
                alpha=alpha,
                colors=mask_color
            )
            
            frame = self.draw_detections(
                image=frame,
                output=XaiDetrObjectDetectionOutput.from_items([d_t, d]),
                colors=colors,
                together=True
            )
            frames.append(frame)
        
        return frames
