from typing import Callable, Dict, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import torch
import transformers as hf
import transformers.models.detr.modeling_detr as detr

from xai_transformers.detr.visualizer import DetrVisualizer
from xai_transformers.detr.core import change_detr_behaviour, XaiDetrObjectDetectionOutput, XaiDetectionItem
from .utils import PerturbationDataset, similarity


class TokenPerturbationEvaluator:
    def __init__(self, model: detr.DetrForObjectDetection, processor: hf.DetrImageProcessor):
        self.model = model
        self.processor = processor
        change_detr_behaviour(self.model)

    @torch.no_grad()
    def _extract_features(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        return self.model.model.extract_features(
            pixel_values=inputs['pixel_values'].to(self.device),
            pixel_mask=inputs['pixel_mask'].to(self.device),
        )

    def _get_collate_fn(self, inputs_mask):
        return lambda masks: inputs_mask.expand(len(masks), -1) * torch.stack(masks).squeeze(1).flatten(1, 2).to(self.device)

    def _get_dataset(self, d: XaiDetectionItem):
        return PerturbationDataset(
            relevance_map=d.relevance_map,
            mode=self.mode,
            percent_per_step=self.percent_per_step,
            out_shape=d.relevance_map.shape,
            device='cpu'
        )

    def _compute_similarity(self, d: XaiDetectionItem, output: detr.DetrObjectDetectionOutput):
        return similarity(
            dt={'bbox': d.pred_box.to(self.device), 'class_prob': d.logits.to(self.device)},
            dp={'bbox': output.pred_boxes, 'class_prob': output.logits}
        )


    def _generate_frames(self, image, perturbed_masks, d_t, output, query_ids, ):
        h, w = d_t.relevance_map.shape
        perturbed_masks = perturbed_masks.view(-1, h, w).float()
        
        ds = [
            XaiDetectionItem(
                logits=output.logits[step, id], 
                pred_box=output.pred_boxes[step, id]
            )
            for step, id in enumerate(query_ids)
        ]
        
        return self.visualizer.generate_frames_for_token_evaluation_v2(
            image=image,
            d_t=d_t,
            d_p=ds,
            masks=perturbed_masks,
            alpha=self.alpha,
            colors=[self.d_t_color, self.d_p_color],
            mask_color=self.mask_color
        )

    def _calculate_metrics(self, results: dict):
        scores = torch.FloatTensor(results['scores']).to(self.device)
        steps = torch.arange(start=1, end=len(results['scores']) + 1, device=self.device, dtype=torch.float) / len(results)
        
        return {
            'score': torch.trapz(scores, steps).item(),
            'scores': results['scores'],
            'iou_scores': results['iou_scores'],
            'cosine_scores': results['cosine_scores'],
            'percent_per_step': self.percent_per_step
        }
    
    def evaluate(
        self,
        image: Image.Image,
        xai_output: XaiDetrObjectDetectionOutput,
        mode: str = 'insertion',
        percent_per_step: float = 0.1,
        batch_size: int = 1,
        device: str = 'cpu',
        verbose: bool = True,
        show_progress: bool = False,
        visualizer: Optional[DetrVisualizer] = None,
        d_t_color: tuple = (0, 0, 255),
        d_p_color: tuple = (255, 0, 0),
        mask_color: tuple = (0, 0, 0),
        alpha: float = 0.7
    ):
        assert show_progress and visualizer is not None, 'visualizer must be provided if show_progress is True'
        
        self.percent_per_step = percent_per_step
        self.mode = mode
        self.batch_size = batch_size
        
        self.visualizer = visualizer
        self.show_progress = show_progress
        self.d_t_color = d_t_color
        self.d_p_color = d_p_color
        self.mask_color = mask_color
        self.alpha = alpha
        
        self.device = device
        self.model = self.model.to(self.device)
        xai_output = xai_output.to(self.device)
        
        self.verbose = verbose
        
        return self._evaluate(image=image, xai_output=xai_output)
    
    @torch.no_grad()
    def _inference(self, inputs_embeds, inputs_mask, object_queries):
        batch_size = inputs_mask.shape[0]
        return self.model(
            inputs_embeds=inputs_embeds.expand(batch_size, -1, -1),
            inputs_mask=inputs_mask,
            object_queries=object_queries.expand(batch_size, -1, -1)
        )
    
    def _evaluate(
        self,
        image: Image.Image,
        xai_output: XaiDetrObjectDetectionOutput
    ):
        inputs_embeds, inputs_mask, object_queries = self._extract_features(image)
        collate_fn = self._get_collate_fn(inputs_mask)
        viz_results = {}

        for item in xai_output:
            results, frames = {'scores': [], 'iou_scores': [], 'cosine_scores': []}, []
            data_loader = DataLoader(
                dataset=self._get_dataset(item),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            for perturbed_masks in tqdm(data_loader, leave=False, disable=not self.verbose):
                output = self._inference(
                    inputs_embeds=inputs_embeds,
                    inputs_mask=perturbed_masks,
                    object_queries=object_queries
                )
                
                scores_step = self._compute_similarity(item, output)
                
                for k in results.keys():
                    results[k].extend(scores_step.pop(k).tolist())

                if self.show_progress:
                    frames.extend(
                        self._generate_frames(
                            image=image, 
                            perturbed_masks=perturbed_masks,
                            d_t=item, 
                            output=output, 
                            query_ids=scores_step.pop('query_ids'),
                        )
                    )
            
            query_id = item.query_id.item()
            xai_output.update_metrics(
                query_id=query_id, 
                name=self.mode, 
                results=self._calculate_metrics(results)
            )

            if self.show_progress:
                viz_results[query_id] = torch.stack(frames)

        return viz_results if self.show_progress else None
