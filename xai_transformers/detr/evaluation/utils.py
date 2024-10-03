import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def iou(bbox_t, bboxes_p):
    x1 = torch.max(bbox_t[0] - bbox_t[2] / 2, bboxes_p[:, :, 0] - bboxes_p[:, :, 2] / 2)
    y1 = torch.max(bbox_t[1] - bbox_t[3] / 2, bboxes_p[:, :, 1] - bboxes_p[:, :, 3] / 2)
    x2 = torch.min(bbox_t[0] + bbox_t[2] / 2, bboxes_p[:, :, 0] + bboxes_p[:, :, 2] / 2)
    y2 = torch.min(bbox_t[1] + bbox_t[3] / 2, bboxes_p[:, :, 1] + bboxes_p[:, :, 3] / 2)
    
    int_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    bbox_t_area = bbox_t[2] * bbox_t[3]
    bboxes_p_area = bboxes_p[:, :, 2] * bboxes_p[:, :, 3]
    union_area = bbox_t_area + bboxes_p_area - int_area

    return int_area / union_area

def cosine_similarity(logits_t, logits_p):
    return F.cosine_similarity(
        logits_t.softmax(-1).unsqueeze(0).unsqueeze(0), 
        logits_p.softmax(-1), 
        dim=2
    )


def similarity(dt, dp):
    batch_size = dp['bbox'].shape[0]
    iou_scores = iou(dt['bbox'], dp['bbox'])
    cosine_scores = cosine_similarity(dt['class_prob'], dp['class_prob'])
    scores, indices = torch.max(iou_scores * cosine_scores, dim=1)
    
    return {
        'scores': scores,
        'iou_scores': iou_scores[range(batch_size), indices],
        'cosine_scores': cosine_scores[range(batch_size), indices],
        'query_ids': indices
    }


def get_sorted_pixel_coordinates(relevance_map: torch.FloatTensor, descending: bool = True):
    H, W = relevance_map.shape
    x = torch.arange(W, device=relevance_map.device).repeat(H)
    y = torch.arange(H, device=relevance_map.device).unsqueeze(1).repeat(1, W).view(-1)
    
    sorted_idx = torch.argsort(relevance_map.view(-1), descending=descending)

    return torch.stack((y[sorted_idx], x[sorted_idx]), dim=1)


class PerturbationDataset(Dataset):

    def __init__(
        self,
        relevance_map: torch.Tensor,
        out_shape: tuple,
        mode: str = 'insertion',
        percent_per_step: float = 0.1,
        device: str = 'cpu',
    ):
        self.out_shape = out_shape
        self.mode = mode
        self.device = device
        self.percent_per_step = percent_per_step

        rm = F.interpolate(
            relevance_map.unsqueeze(0).unsqueeze(0),
            size=out_shape, mode='bilinear'
        ).squeeze(0).squeeze(0).to(self.device)
        
        self.sorted_coords = get_sorted_pixel_coordinates(rm, descending=True).to(self.device)

        self.total_pixels = self.sorted_coords.shape[0]
        self.pixels_per_step = int(self.percent_per_step * self.total_pixels)
        cumulative_mask = torch.zeros((1, *out_shape), dtype=torch.bool, device=self.device)
        n = self.__len__()

        if self.mode == 'insertion':
            self.masks = torch.zeros((n, 1, *out_shape), dtype=torch.bool, device=self.device)
        else:
            self.masks = torch.ones((n, 1, *out_shape), dtype=torch.bool, device=self.device)
        
        for idx in range(n):
            start_idx = idx * self.pixels_per_step
            end_idx = min((idx + 1) * self.pixels_per_step, self.total_pixels)

            i = self.sorted_coords[start_idx: end_idx, 0]
            j = self.sorted_coords[start_idx: end_idx, 1]
            cumulative_mask[0, i, j] = True
            
            if self.mode == 'insertion':
                self.masks[idx] |= cumulative_mask
            else:
                self.masks[idx] &= ~cumulative_mask

    def __len__(self):
        return self.total_pixels // self.pixels_per_step

    def __getitem__(self, idx):
        return self.masks[idx].clone()
