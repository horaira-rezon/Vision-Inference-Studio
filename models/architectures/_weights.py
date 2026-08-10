import torch


def safe_torch_load(weight_path):
    try:
        return torch.load(weight_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(weight_path, map_location="cpu", weights_only=False)


def extract_state_and_names(checkpoint):
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    metadata_names = checkpoint.get("class_names", checkpoint.get("names")) if isinstance(checkpoint, dict) else None
    if isinstance(metadata_names, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(metadata_names)}
    elif isinstance(metadata_names, dict):
        names = {int(k): str(v) for k, v in metadata_names.items()}
    else:
        names = None
    return state, names
