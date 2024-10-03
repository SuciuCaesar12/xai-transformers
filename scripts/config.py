from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union

class ModelConfig(BaseModel):
    pretrained_model_name_or_path: str
    pretrained_processor_name_or_path: str
    id2label: Dict[int, str]

class FlagsConfig(BaseModel):
    enable_eval: bool
    override_explanations: bool
    override_evaluations: bool

class ExplainerConfig(BaseModel):
    score_threshold: float
    area_threshold: float
    include_labels: Union[List[str], str] = 'all'

class EvaluationConfig(BaseModel):
    mode: str  
    percent_per_step: float
    batch_size: int
    show_progress: bool
    
    d_t_color: tuple = (0, 0, 255)
    d_p_color: tuple = (255, 0, 0)
    mask_color: tuple = (0, 0, 0)
    alpha: float = 0.7

class PathsConfig(BaseModel):
    images_dir: str
    output_dir: str

class Config(BaseModel):
    defaults: list = Field(default_factory=lambda: ["_self_"])
    device: str
    flags: FlagsConfig
    model: ModelConfig
    explainer: ExplainerConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
