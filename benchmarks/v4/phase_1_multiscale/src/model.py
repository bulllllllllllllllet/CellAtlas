from __future__ import annotations

from pathlib import Path
import torch
from torchvision.models.segmentation import deeplabv3_resnet50


def build_model(config: dict, num_classes: int, input_channels: int = 3) -> torch.nn.Module:
    path=config["model"].get("pretrained_weights")
    if not path: raise ValueError("model.pretrained_weights must name a local checkpoint; random initialization is intentionally not supported")
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f"configured pretrained weight file does not exist: {path}")
    model=deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False)
    state=torch.load(path,map_location="cpu",weights_only=False)
    # Public torchvision checkpoints use ``state_dict`` while our Phase 1
    # artifacts store learned weights under ``model``.  Accept both so later
    # modules can initialise from the selected Phase 1 checkpoint.
    state=state.get("model",state.get("state_dict",state))
    clean={k.removeprefix("module."):v for k,v in state.items()}
    # The public COCO checkpoint has a 21-class head. Retain every tensor whose
    # name and shape match; the 12-class heads are deliberately trained here.
    target=model.state_dict(); compatible={k:v for k,v in clean.items() if k in target and target[k].shape == v.shape}
    if not any(k.startswith("backbone.") for k in compatible):
        raise ValueError("checkpoint contains no compatible DeepLabV3-ResNet50 backbone tensors")
    model.load_state_dict(compatible,strict=False)
    if input_channels != 3:
        old=model.backbone.conv1; new=torch.nn.Conv2d(input_channels,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        with torch.no_grad(): new.weight.copy_(old.weight.repeat(1,input_channels//3,1,1)/(input_channels//3))
        model.backbone.conv1=new
    return model
