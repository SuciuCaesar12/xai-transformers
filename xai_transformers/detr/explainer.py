from .core import XaiDetrObjectDetectionOutput
import xai_transformers.rules as xai_utils

import transformers.models.detr.modeling_detr as detr
import transformers as hf

from PIL import Image
import torch
import tqdm
from typing import Dict, List, Optional, Tuple, Union


DUMMY_ANNOTATIONS = {'image_id': -1, 'annotations': []}


def normalize(x: torch.Tensor, min_v: float = 0., max_v: float = 1.) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()) * (max_v - min_v) + min_v


class DetrExplainer:
    
    def __init__(self, model: detr.DetrForObjectDetection, processor: hf.DetrImageProcessor, device: str = 'cpu'):
        self.model = model
        self.processor = processor
        self.device = device

    def _setup_labels(self, include_labels: Optional[List[str]] = None):
        label2id = self.model.config.label2id
        if include_labels == 'all':
            self.include_label_ids = torch.tensor(list(label2id.values())).to(self.device)
        else:
            self.include_label_ids = torch.tensor(list(map(label2id.get, include_labels))).to(self.device)
        self.include_labels = include_labels

    def _enable_grads(self):
        self.model.requires_grad_(True)
        self.model.model.freeze_backbone()
    
    def preprocess(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        return self.processor(
            images=image,
            annotations=DUMMY_ANNOTATIONS,
            return_tensors="pt"
        )

    def _filter_outputs(self, outputs: XaiDetrObjectDetectionOutput) -> XaiDetrObjectDetectionOutput:
        keep = torch.ones(outputs.logits.shape[1], dtype=torch.bool, device=self.device)
        filter_by_score = self.score_threshold is not None
        filter_by_area = self.area_threshold is not None
        filter_by_labels = self.include_labels != 'all'
        
        if filter_by_score or filter_by_labels:
            scores, label_ids = outputs.logits.squeeze(0).softmax(-1)[..., :-1].max(-1)
        
        if filter_by_score:
            keep &= scores >= self.score_threshold
        
        if filter_by_area:
            h, w = outputs.pred_boxes.squeeze(0)[..., 2:].unbind(-1)
            keep &= h * w >= self.area_threshold
        
        if filter_by_labels:
            keep &= keep & torch.isin(label_ids, self.include_label_ids.to(self.device))
        
        return keep.nonzero().squeeze(1)
        
    def _inference(self, inputs: Dict[str, torch.Tensor]) -> Tuple[XaiDetrObjectDetectionOutput, torch.Tensor, torch.Tensor]:
        features = []
        hook = self.model.model.backbone.conv_encoder.register_forward_hook(
            lambda m, i, o: features.append(tuple(reversed([f[0] for f in o])))
        )
        
        self.model = self.model.to(self.device)
        self._enable_grads()
        outputs = self.model(
            pixel_values=inputs['pixel_values'].to(self.device),
            pixel_mask=inputs['pixel_mask'].to(self.device),
            output_attentions=True,
            output_hidden_states=False,
        )
        outputs = XaiDetrObjectDetectionOutput.from_detr_object_detection_output(outputs)
        hook.remove()
        
        filtered_query_ids = self._filter_outputs(outputs)
        rel_map_shape = features.pop()[0].shape[2:]
        
        return outputs, filtered_query_ids, rel_map_shape
    
    def explain_image(
        self,
        image: Optional[Image.Image] = None,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        include_labels: Optional[Union[List[str], str]] = None,
        score_threshold: Optional[float] = None,
        area_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> XaiDetrObjectDetectionOutput:
        assert image is not None or inputs is not None, "Either image or inputs must be provided"
        
        if device is not None:
            self.device = device
        if score_threshold is not None:
            self.score_threshold = score_threshold
        if area_threshold is not None:
            self.area_threshold = area_threshold
        if include_labels is not None:
            self._setup_labels(include_labels)

        return self._explain_image(image, inputs)
    
    def _initialize_rel_matrices(self, N_i: int, N_q: int):
        R_i_i = torch.eye(N_i, device=self.device)
        R_q_q = torch.eye(N_q, device=self.device)
        R_q_i = torch.zeros(N_q, N_i, device=self.device)
        return R_i_i, R_q_q, R_q_i
    
    def _backward(self, idx: int, logits: torch.Tensor):
        idx = idx.item()
        one_hot = torch.zeros_like(logits[idx], dtype=torch.float)
        one_hot[logits[idx].argmax().item()] = 1.
        one_hot.requires_grad = True

        loss = (logits[idx] * one_hot).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _generate_rel_maps(
        self,
        outputs: XaiDetrObjectDetectionOutput,
        query_ids,
        out_shape: Tuple[int, int],
        verbose: bool = False
    ) -> Optional[torch.Tensor]:
        N_i = outputs.encoder_attentions[0].shape[-1]  # Number of image tokens
        N_q = self.model.config.num_queries            # Number of query tokens
        logits = outputs.logits.squeeze(0)
        logits.requires_grad_(True)
        rel_maps = []

        for am in outputs.encoder_attentions + outputs.decoder_attentions + outputs.cross_attentions:
            am.requires_grad_(True)
            am.retain_grad()

        for q_i in tqdm.tqdm(query_ids, desc='Generating relevance maps', total=len(query_ids), disable=not verbose):
            R_i_i, R_q_q, R_q_i = self._initialize_rel_matrices(N_i, N_q)
            self._backward(q_i, logits)

            for am in outputs.encoder_attentions:
                xai_utils.enc_self_attn_update(am=am, R_e_e=R_i_i)

            for dec_am, crs_am in zip(outputs.decoder_attentions, outputs.cross_attentions):
                xai_utils.dec_self_attn_update(
                    dec_am, R_d_d=R_q_q, R_d_e=R_q_i
                )
                xai_utils.dec_crs_attn_update(
                    crs_am, R_d_d=R_q_q, R_e_e=R_i_i, R_d_e=R_q_i
                )

            rel_maps.append(normalize(R_q_i[q_i]).clone())

        if len(rel_maps):
            return torch.stack(rel_maps).view(-1, *out_shape).unsqueeze(0)
        else:
            return None
    
    def _explain_image(
        self, 
        image: Optional[Image.Image] = None, 
        inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> XaiDetrObjectDetectionOutput:
        if inputs is None:
            inputs = self.preprocess(image)
        
        outputs, query_ids, out_shape = self._inference(inputs)
        
        rel_maps = self._generate_rel_maps(outputs, query_ids, out_shape)
        
        outputs = outputs.detach().select(query_ids)
        outputs.__setitem__('relevance_maps', rel_maps)
        
        return outputs.squeeze()