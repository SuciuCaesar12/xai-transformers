from pathlib import Path
import sys
from typing import Any, Dict

from .visualizer import DetrVisualizer
from .core import XaiDetrObjectDetectionOutput, XaiDetectionItem

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torch


class TensorboardLogger:
    
    def __init__(
        self,
        log_dir: Path,
        visualizer: DetrVisualizer,
    ):
        self.summary_writer = SummaryWriter(log_dir)
        self.visualizer = visualizer

    def log(
        self,
        image: Image.Image,
        outputs: XaiDetrObjectDetectionOutput,
        viz_results: Dict[str, Any] = None
    ):
        results = self.visualizer.draw_detections_and_relevance_maps(image, outputs)
        
        for item, result in zip(outputs, results):
            item: XaiDetectionItem
            query_id = item.query_id.item()
            tag = f'/{query_id}'
            
            self.summary_writer.add_image(f'{tag}/detection_and_relevance_map', result)
            
            if item.metrics is not None:
                for mode, results in item.metrics.items():
                    steps = torch.arange(start=1, end=len(results['scores']) + 1) 
                    steps = (steps * (results['percent_per_step'] * 100.)).int()
                    
                    for idx, step in enumerate(steps):
                        self.summary_writer.add_scalars(
                            main_tag=f'{tag}/{mode}',
                            tag_scalar_dict={
                                'scores': results['scores'][idx],
                                'iou_scores': results['iou_scores'][idx],
                                'cosine_scores': results['cosine_scores'][idx]
                            },
                            global_step=step
                        )
        
                if viz_results is not None:
                    for query_id, frames in viz_results.items():
                        for step, frame in enumerate(frames):
                            self.summary_writer.add_image(
                                tag=f'{query_id}/{mode}',
                                img_tensor=frame,
                                global_step=step
                            )
