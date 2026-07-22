from __future__ import annotations

import torch
import torch.nn.functional as F

from benchmarks.v4.phase_1_multiscale.src.metrics import soft_dice_loss


def _semantic_boundary(mask: torch.Tensor, ignore_index: int) -> torch.Tensor:
    valid = mask.ne(ignore_index)
    edge = torch.zeros_like(valid, dtype=torch.bool)
    edge[:, 1:] |= (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[:, :, 1:] |= (mask[:, :, 1:] != mask[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    return edge.float()


def assignment_boundary(assignment: torch.Tensor) -> torch.Tensor:
    """Differentiable region edge probability from neighbouring assignments."""
    horizontal = 1.0 - (assignment[:, :, :, 1:] * assignment[:, :, :, :-1]).sum(1)
    vertical = 1.0 - (assignment[:, :, 1:, :] * assignment[:, :, :-1, :]).sum(1)
    # No in-place writes: this tensor participates in the region-loss graph.
    horizontal = F.pad(horizontal, (1, 0, 0, 0))
    vertical = F.pad(vertical, (0, 0, 1, 0))
    return torch.maximum(horizontal, vertical).clamp(1e-5, 1 - 1e-5)


class RegionizationLoss(torch.nn.Module):
    def __init__(self, num_classes: int, ignore_index: int, background_ids: set[int], weights: dict[str, float]):
        super().__init__()
        self.num_classes, self.ignore, self.background, self.weights = num_classes, ignore_index, background_ids, weights

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor, slic: torch.Tensor, slic_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
        full_target = target
        low_size = output["assignment_low"].shape[-2:]
        target = F.interpolate(target[:, None].float(), size=low_size, mode="nearest")[:, 0].long()
        slic = F.interpolate(slic[:, None].float(), size=low_size, mode="nearest")[:, 0].long()
        # Region losses are evaluated in float32 at backbone resolution. This
        # avoids both the 512x512 one-hot allocation and float16 log(0) NaNs.
        with torch.autocast(device_type=output["assignment_low"].device.type, enabled=False):
            assignment = output["assignment_low"].float()
            valid = target.ne(self.ignore)
            if int(slic.max().item()) >= assignment.shape[1]: raise ValueError(f"SLIC label {int(slic.max().item())} exceeds {assignment.shape[1]} assignment slots")
            # Dataset canonicalises labels by centroid after augmentation, so
            # slot IDs are spatially consistent and direct NLL prevents the
            # degenerate all-regions-to-one-slot solution.
            slic_nll = -assignment.clamp_min(1e-4).log().gather(1,slic.unsqueeze(1)).squeeze(1)
            slic_loss = (slic_nll * valid).sum() / valid.sum().clamp_min(1)
            boundary_prob = assignment_boundary(assignment).clamp(1e-4, 1.0 - 1e-4)
            semantic_boundary = _semantic_boundary(target, self.ignore).float()
            slic_boundary = _semantic_boundary(slic, -1).float()
            # Warmup follows SLIC internal edges; semantic boundaries are
            # always retained. SLIC influence then decays with distillation.
            boundary_target = torch.maximum(semantic_boundary, slic_boundary * float(slic_weight))
            boundary = -(boundary_target * boundary_prob.log() + (1.0 - boundary_target) * (1.0 - boundary_prob).log())
            boundary_loss = (boundary * valid).sum() / valid.sum().clamp_min(1)
            valid_assignment = assignment * valid.unsqueeze(1)
            mass_px = valid_assignment.sum((2, 3)).clamp_min(1e-6)
            mass = mass_px / valid.sum((1, 2)).unsqueeze(1).clamp_min(1)
            # KL(mass || uniform) is O(log K) under collapse; the previous
            # mean squared term was only O(1/K) and could not prevent empty slots.
            balance = (mass * (mass * assignment.shape[1]).clamp_min(1e-6).log()).sum(1).mean()
            h, w = assignment.shape[-2:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=assignment.device), torch.linspace(-1, 1, w, device=assignment.device), indexing="ij")
            cx = (valid_assignment * xx).sum((2, 3)) / mass_px; cy = (valid_assignment * yy).sum((2, 3)) / mass_px
            distance = (xx[None, None] - cx[:, :, None, None]).square() + (yy[None, None] - cy[:, :, None, None]).square()
            compact_per_region = (valid_assignment * distance).sum((2, 3)) / mass_px
            compact = (compact_per_region * mass).sum(1).mean()
            safe = target.masked_fill(~valid, 0)
            classes = F.one_hot(safe, self.num_classes).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
            per_region_class = torch.einsum("bkhw,bchw->bkc", valid_assignment, classes)
            probs = per_region_class / per_region_class.sum(-1, keepdim=True).clamp_min(1e-6)
            entropy = -(probs * probs.clamp_min(1e-6).log()).sum(-1)
            purity = (entropy * mass).sum(1).mean()
            semantic_logits = output["semantic_logits"].float()
            ce = F.cross_entropy(semantic_logits, full_target, ignore_index=self.ignore)
            dice = soft_dice_loss(semantic_logits, full_target, self.num_classes, self.ignore, self.background)
        total = (self.weights["slic"] * slic_weight * slic_loss + self.weights["boundary"] * boundary_loss + self.weights["balance"] * balance + self.weights["compact"] * compact + self.weights["purity"] * purity + self.weights["semantic_ce"] * ce + self.weights["semantic_dice"] * dice)
        parts = {"loss": float(total.detach()), "slic": float(slic_loss.detach()), "boundary": float(boundary_loss.detach()), "balance": float(balance.detach()), "compact": float(compact.detach()), "purity": float(purity.detach()), "semantic_ce": float(ce.detach()), "semantic_dice": float(dice.detach())}
        return total, parts
