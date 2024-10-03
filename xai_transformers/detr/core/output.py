from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import transformers.models.detr.modeling_detr as detr
import torch

FIELDS = [
    'logits',
    'pred_boxes',
    'encoder_attentions',
    'decoder_attentions',
    'cross_attentions'
]


@dataclass
class XaiDetectionItem:
    query_id: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    pred_box: Optional[torch.Tensor] = None
    relevance_map: Optional[torch.Tensor] = None
    metrics: Optional[dict] = None
    
    
    def __str__(self) -> str:
        string = 'XaiDetectionItem(\n'
        if self.query_id is not None:
            string += f'  query_id: {self.query_id.tolist()}\n'
        string += f'  logits: Tensor with shape {tuple(self.logits.shape)}\n'
        string += f'  pred_box: Tensor with shape {tuple(self.pred_box.shape)}\n'
        if self.relevance_map is not None:
            string += f'  relevance_map: Tensor with shape {tuple(self.relevance_map.shape)}\n'
        if self.metrics is not None:
            string += f'  metrics: {list(self.metrics.keys())}\n'
        string += ')\n'
        return string.strip()


@dataclass
class XaiDetrObjectDetectionOutput(detr.DetrObjectDetectionOutput):
    query_ids: Optional[torch.Tensor] = None
    relevance_maps: Optional[torch.FloatTensor] = None
    metrics: Dict[str, Any] = None
    
    @classmethod
    def from_detr_object_detection_output(cls, output: detr.DetrObjectDetectionOutput) -> "XaiDetrObjectDetectionOutput":
        n_queries = output.pred_boxes.shape[1]
        device = output.pred_boxes.device            
        
        return XaiDetrObjectDetectionOutput(
            query_ids=torch.arange(n_queries, device=device).unsqueeze(0), 
            relevance_maps=None,
            **{name: getattr(output, name) for name in FIELDS}
        )
    
    @classmethod
    def from_items(cls, items: List[XaiDetectionItem]) -> "XaiDetrObjectDetectionOutput":
        return XaiDetrObjectDetectionOutput(
            logits=torch.stack([i.logits for i in items]),
            pred_boxes=torch.stack([i.pred_box for i in items]),
        )
            

    def empty(self) -> bool:
        return self.query_ids.numel() == 0

    def is_batched(self) -> bool:
        return self.logits.ndim == 3
    
    def _apply_op(self, op: Callable, exclude: List[str] = []) -> "XaiDetrObjectDetectionOutput":
        for field in fields(self):
            name = field.name
            if name in exclude:
                continue
            value = getattr(self, name)
            
            if value is not None:
                if isinstance(value, torch.Tensor):
                    setattr(self, name, op(value))
                
                if isinstance(value, tuple) and all(isinstance(v, torch.Tensor) for v in value):
                    setattr(self, name, tuple(op(v) for v in value))
        
        return self
    
    def to(self, device: str) -> "XaiDetrObjectDetectionOutput":
        return self._apply_op(lambda x: x.to(device))
    
    def detach(self) -> "XaiDetrObjectDetectionOutput":
        return self._apply_op(lambda x: x.detach())

    def unsqueeze(self, dim: int = 0) -> "XaiDetrObjectDetectionOutput":
        return self._apply_op(lambda x: x.unsqueeze(dim))
    
    def squeeze(self, dim: int = 0) -> "XaiDetrObjectDetectionOutput":
        return self._apply_op(lambda x: x.squeeze(dim))
    
    def select(self, query_ids: torch.Tensor) -> "XaiDetrObjectDetectionOutput":
        if query_ids.ndim == 1:
            query_ids = query_ids.unsqueeze(0)
        indices = torch.nonzero(self.query_ids.view(-1, 1) == query_ids, as_tuple=False)[:, 0]
        
        return self._apply_op(lambda x: x[:, indices], exclude=[f for f in FIELDS if f.endswith('_attentions')])
    
    def __str__(self) -> str:
        string = 'XaiDetrObjectDetectionOutput(\n'
        
        if self.query_ids is not None:
            string += f'  query_ids: {self.query_ids.tolist()}\n'
        if self.relevance_maps is not None:
            string += f'  relevance_maps: Tensor with shape {tuple(self.relevance_maps.shape)}\n'
            
        string += f'  logits: Tensor with shape {tuple(self.logits.shape)}\n'
        string += f'  pred_boxes: Tensor with shape {tuple(self.pred_boxes.shape)}\n'
        
        if self.encoder_attentions is not None:
            string += f'  encoder_attentions: {len(self.encoder_attentions)} tensors with shapes {tuple(self.encoder_attentions[0].shape)}\n'
        if self.decoder_attentions is not None:
            string += f'  decoder_attentions: {len(self.decoder_attentions)} tensors with shapes {tuple(self.decoder_attentions[0].shape)}\n'
        if self.cross_attentions is not None:
            string += f'  cross_attentions: {len(self.cross_attentions)} tensors with shapes {tuple(self.cross_attentions[0].shape)}\n'
        
        string += ')\n'
        return string.strip()

    
    def update_metrics(self, query_id: int, name: str, results: Any) -> "XaiDetrObjectDetectionOutput":
        if self.metrics is None:
            self.metrics = {query_id: {name: results}}
        elif query_id not in self.metrics:
            self.metrics[query_id] = {name: results}
        elif name not in self.metrics[query_id]:
            self.metrics[query_id][name] = results
        
        return self
    
    
    def to_dict(
        self,
        keep_attentions: bool = False,
        keep_relevance_maps: bool = True,
        keep_metrics: bool = True,
        copy: bool = True
    ) -> dict:
        state_dict = {
            'query_ids': self.query_ids,
            'logits': self.logits,
            'pred_boxes': self.pred_boxes
        }

        if keep_attentions:
            for f in fields(self):
                if f.name.endswith('_attentions'):
                    value = getattr(self, f.name)
                    if value is not None:
                        state_dict[f.name] = value

        if keep_relevance_maps and self.relevance_maps is not None:
            state_dict['relevance_maps'] = self.relevance_maps
        
        if keep_metrics and self.metrics is not None:
            state_dict['metrics'] = self.metrics
        
        return deepcopy(state_dict) if copy else state_dict

    def save(
        self, 
        filepath: Path, 
        keep_attentions: bool = False,
        keep_relevance_maps: bool = True,
        keep_metrics: bool = True,
    ):
        state_dict = self.to_dict(
            keep_attentions=keep_attentions,
            keep_relevance_maps=keep_relevance_maps,
            keep_metrics=keep_metrics
        )

        torch.save(state_dict, filepath)

    @classmethod
    def load(cls, filepath: Path) -> 'XaiDetrObjectDetectionOutput':
        state_dict: dict = torch.load(filepath)
        loaded_output = cls.__new__(cls)

        for attr, value in state_dict.items():
            setattr(loaded_output, attr, value)

        return loaded_output
    
    def __get__(self, index: int) -> XaiDetectionItem:
        query_id = self.query_ids[index]
        relevance_map = self.relevance_maps[index] if self.relevance_maps is not None else None
        metrics = self.metrics.get(query_id.item(), None) if self.metrics is not None else None
        
        return XaiDetectionItem(
            query_id=self.query_ids[index],
            relevance_map=relevance_map,
            logits=self.logits[index],
            pred_box=self.pred_boxes[index],
            metrics=metrics
        )


    def __iter__(self) -> Iterator[XaiDetectionItem]:
        for i in range(len(self.query_ids)):
            yield self.__get__(i)


