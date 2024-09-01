import torch


def set_device(device) -> torch.device:
    if device in ["cpu", "cuda", "mps"]:
        try:
            device = torch.device(device)
            torch.tensor([[0, 3], [5, 7]], dtype=torch.float32, device=device)
        except Exception as e:
            print(e)
            device = torch.device("cpu")
            print("Switched to CPU")
    else:
        device = torch.device("cpu")

    return device
