from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from PIL import Image
from tqdm import tqdm

import transformers as hf
import xai_transformers.detr as xai_detr
from config import Config

hf.logging.set_verbosity_error()


def process_image(
    image_path: Path, 
    explainer: xai_detr.DetrExplainer, 
    evaluator: xai_detr.TokenPerturbationEvaluator, 
    visualizer: xai_detr.DetrVisualizer,
    cfg: Config
):
    exp_filepath = Path(cfg.paths.output_dir) / image_path.stem / 'explanations.pkl'
    
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    output = None
    if exp_filepath.exists() and not cfg.flags.override_explanations:
        output = xai_detr.XaiDetrObjectDetectionOutput.load(filepath=exp_filepath)

    if output is None:
        output = explainer.explain_image(
            image=image, 
            device=cfg.device,
            **cfg.explainer.model_dump()
        )

    if output.empty():
        return

    if cfg.flags.enable_eval:
        log_dir = Path(cfg.paths.output_dir) / image_path.stem / 'logs'
        if log_dir.exists():
            if cfg.flags.override_evaluations:
                for item in log_dir.iterdir():
                    item.unlink()
            else:
                return
        
        logger = xai_detr.TensorboardLogger(
            log_dir=Path(cfg.paths.output_dir) / image_path.stem / 'logs',
            visualizer=visualizer
        )
        
        eval_visualizations = evaluator.evaluate(
            image=image, 
            xai_output=output, 
            device=cfg.device, 
            verbose=False, 
            visualizer=visualizer,
            **cfg.evaluation.model_dump()
        )

        logger.log(
            image=image,
            outputs=output.to('cpu'), 
            viz_results=eval_visualizations
        )
    
    output.save(filepath=exp_filepath)

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    typed_cfg = Config(**cfg_dict)

    images_dir = Path(typed_cfg.paths.images_dir)
    output_dir = Path(typed_cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = hf.DetrForObjectDetection.from_pretrained(typed_cfg.model.pretrained_model_name_or_path)
    processor = hf.DetrImageProcessor.from_pretrained(typed_cfg.model.pretrained_processor_name_or_path)

    explainer = xai_detr.DetrExplainer(model=model, processor=processor)
    evaluator = xai_detr.TokenPerturbationEvaluator(model=model, processor=processor)
    visualizer = xai_detr.DetrVisualizer(processor=processor, id2label=model.config.id2label)

    image_paths = list(images_dir.glob("*.jpg"))
    for image_path in tqdm(image_paths, desc="Processing images"):
        process_image(
            image_path=image_path, 
            explainer=explainer, 
            evaluator=evaluator, 
            visualizer=visualizer, 
            cfg=typed_cfg
        )

    print("Processing completed for all images.")

if __name__ == "__main__":
    main()
