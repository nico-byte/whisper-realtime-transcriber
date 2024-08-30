import torch

from punctuators.models import PunctCapSegModelONNX
from typing import Tuple, List


def preprocess_text(model: PunctCapSegModelONNX, inputs: str) -> Tuple[List[str], List[str]]:
    inputs_list = []
    inputs_list.append(inputs)

    outputs: List[List[str]] = model.infer(
        texts=inputs_list,
        apply_sbd=True,
    )

    outputs = outputs[0]

    partial_sentence = outputs[-1]
    partial_sentence += " "
    full_sentences = [text for text in outputs[:-1]]
    full_sentences = " ".join(full_sentences)

    return full_sentences, partial_sentence


def set_device(device) -> torch.device:
    if device in ["cpu", "cuda", "mps"]:
        try:
            device = torch.device(device)
            torch.tensor([[0, 3],[5, 7]], dtype=torch.float32, device=device)
        except Exception as e:
            print(e)
            device = torch.device("cpu")
            print("Switched to CPU")
    else:
        device = torch.device("cpu")

    return device
