from transformers.models.detr.modeling_detr import *
from types import MethodType


def forward_DetrModel(
    self: DetrModel,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_mask: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.FloatTensor] = None,
    encoder_outputs: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,  # ADDED -> cached input embeddings
    inputs_mask: Optional[torch.FloatTensor] = None,    # ADDED -> cached mask for input embeddings
    object_queries: Optional[torch.FloatTensor] = None, # ADDED -> cached object queries
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.FloatTensor], DetrModelOutput]:

    # Handling output flags
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # extract features only if tensors are not cached
    if not (inputs_embeds is not None and inputs_mask is not None and object_queries is not None):
        batch_size = pixel_values.shape[0]
        flattened_features, flattened_mask, object_queries = self.extract_features(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask
        )
    else:
        # Use the cached tensors
        batch_size = inputs_embeds.shape[0]
        flattened_features = inputs_embeds
        flattened_mask = inputs_mask

    # Forward pass through encoder
    if encoder_outputs is None:
        encoder_outputs = self.encoder(
            inputs_embeds=flattened_features,
            attention_mask=flattened_mask,
            object_queries=object_queries,  # CACHED or not
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # Handling the decoder inputs and outputs
    query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
    queries = torch.zeros_like(query_position_embeddings)
    decoder_outputs = self.decoder(
        inputs_embeds=queries,
        attention_mask=None,
        object_queries=object_queries,
        query_position_embeddings=query_position_embeddings,
        encoder_hidden_states=encoder_outputs[0],
        encoder_attention_mask=flattened_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return DetrModelOutput(
        last_hidden_state=decoder_outputs.last_hidden_state,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
        intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
    )


def forward_DetrForObjectDetection(
    self: DetrForObjectDetection,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_mask: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.FloatTensor] = None,
    encoder_outputs: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,  # ADDED -> cached input embeddings
    inputs_mask: Optional[torch.FloatTensor] = None,    # ADDED -> cached mask for input embeddings
    object_queries: Optional[torch.FloatTensor] = None, # ADDED -> cached object queries
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[List[dict]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.FloatTensor], DetrObjectDetectionOutput]:
    r"""
    labels (`List[Dict]` of len `(batch_size,)`, *optional*):
        Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
        following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
        respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
        in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, DetrForObjectDetection
    >>> import torch
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    >>> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    >>> inputs = image_processor(images=image, return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    >>> target_sizes = torch.tensor([image.size[::-1]])
    >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    ...     0
    ... ]

    >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    ...     box = [round(i, 2) for i in box.tolist()]
    ...     print(
    ...         f"Detected {model.config.id2label[label.item()]} with confidence "
    ...         f"{round(score.item(), 3)} at location {box}"
    ...     )
    Detected remote with confidence 0.998 at location [40.16, 70.81, 175.55, 117.98]
    Detected remote with confidence 0.996 at location [333.24, 72.55, 368.33, 187.66]
    Detected couch with confidence 0.995 at location [-0.02, 1.15, 639.73, 473.76]
    Detected cat with confidence 0.999 at location [13.24, 52.05, 314.02, 470.93]
    Detected cat with confidence 0.999 at location [345.4, 23.85, 640.37, 368.72]
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # First, sent images through DETR base model to obtain encoder + decoder outputs
    outputs = self.model(
        pixel_values,
        pixel_mask=pixel_mask,
        decoder_attention_mask=decoder_attention_mask,
        encoder_outputs=encoder_outputs,
        inputs_embeds=inputs_embeds,    ## CACHED or not
        inputs_mask=inputs_mask,        ## CACHED or not
        object_queries=object_queries,  ## CACHED or not
        decoder_inputs_embeds=decoder_inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]

    # class logits + predicted bounding boxes
    logits = self.class_labels_classifier(sequence_output)
    pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

    loss, loss_dict, auxiliary_outputs = None, None, None
    if labels is not None:
        # First: create the matcher
        matcher = DetrHungarianMatcher(
            class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
        )
        # Second: create the criterion
        losses = ["labels", "boxes", "cardinality"]
        criterion = DetrLoss(
            matcher=matcher,
            num_classes=self.config.num_labels,
            eos_coef=self.config.eos_coefficient,
            losses=losses,
        )
        criterion.to(self.device)
        # Third: compute the losses, based on outputs and labels
        outputs_loss = {}
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes
        if self.config.auxiliary_loss:
            intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
            outputs_class = self.class_labels_classifier(intermediate)
            outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs

        loss_dict = criterion(outputs_loss, labels)
        # Fourth: compute total loss, as a weighted sum of the various losses
        weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
        weight_dict["loss_giou"] = self.config.giou_loss_coefficient
        if self.config.auxiliary_loss:
            aux_weight_dict = {}
            for i in range(self.config.decoder_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    if not return_dict:
        if auxiliary_outputs is not None:
            output = (logits, pred_boxes) + auxiliary_outputs + outputs
        else:
            output = (logits, pred_boxes) + outputs
        return ((loss, loss_dict) + output) if loss is not None else output

    return DetrObjectDetectionOutput(
        loss=loss,
        loss_dict=loss_dict,
        logits=logits,
        pred_boxes=pred_boxes,
        auxiliary_outputs=auxiliary_outputs,
        last_hidden_state=outputs.last_hidden_state,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
    )


def extract_features(
    self: DetrModel,
    pixel_values: torch.FloatTensor,
    pixel_mask: Optional[torch.LongTensor] = None,
):
    batch_size, num_channels, height, width = pixel_values.shape
    device = pixel_values.device
    
    if pixel_mask is None:
        pixel_mask = torch.ones(((batch_size, height, width)), device=device)

    # First, sent pixel_values + pixel_mask through Backbone to obtain the features
    # pixel_values should be of shape (batch_size, num_channels, height, width)
    # pixel_mask should be of shape (batch_size, height, width)
    features, object_queries_list = self.backbone(pixel_values, pixel_mask)

    # get final feature map and downsampled mask
    feature_map, mask = features[-1]

    if mask is None:
        raise ValueError("Backbone does not return downsampled pixel mask")

    # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
    projected_feature_map = self.input_projection(feature_map)

    # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
    # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
    flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
    object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)

    flattened_mask = mask.flatten(1)
    
    return flattened_features, flattened_mask, object_queries


def change_detr_behaviour(model: DetrForObjectDetection):
    model.model.forward = MethodType(forward_DetrModel, model.model)
    model.model.extract_features = MethodType(extract_features, model.model)
    model.forward = MethodType(forward_DetrForObjectDetection, model)

    return model
