import torch
import torch.nn.functional as F


def avg_heads(am_grad: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
    return (am_grad * am).clamp(0).mean(1).squeeze(0)

def enc_self_attn_update(am: torch.Tensor, R_e_e: torch.Tensor):
    am_grad, am = am.grad.detach(), am.detach()
    cam = avg_heads(am_grad, am)
    R_e_e.add_(cam @ R_e_e)

def dec_self_attn_update(am: torch.Tensor, R_d_d: torch.Tensor, R_d_e: torch.Tensor):
    am_grad, am = am.grad.detach(), am.detach()
    cam = avg_heads(am_grad, am)
    R_d_d.add_(cam @ R_d_d)
    R_d_e.add_(cam @ R_d_e)

def norm_rel_map(rel_map: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(*rel_map.shape).to(rel_map.device)
    return F.normalize(rel_map - eye, p=1, dim=0) + eye

def dec_crs_attn_update(am: torch.Tensor, R_d_d: torch.Tensor, R_e_e: torch.Tensor, R_d_e: torch.Tensor):
    am_grad, am = am.grad.detach(), am.detach()
    cam = avg_heads(am_grad, am)

    norm_R_q_q = norm_rel_map(R_d_d)
    norm_R_i_i = norm_rel_map(R_e_e)
    R_d_e.add_(norm_R_q_q.T @ cam @ norm_R_i_i)
