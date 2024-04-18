import string
import torch
import numpy as np

from typing import Tuple, List
from num2words import num2words
from nltk.tokenize import WordPunctTokenizer


def tokenize_text(text: str, language: str='en') -> Tuple[List[str], List[str]]:
    # Keep original tokens
    original_tokens =  WordPunctTokenizer().tokenize(text)
    
    # Convert text to lower case
    text = text.lower()

    # Convert numbers to words
    # ordinal numbers will be converted to usual numbers - 2nd will be two
    text = ' '.join(num2words(int(word), lang=language, ordinal=False) if word.isdigit() else word for word in text.split())
    
    # Keep German umlauts
    remove_punct_map = {ord(char): None for char in string.punctuation if char not in ['ä', 'ö', 'ü', 'ß']}
    
    # Remove punctuation and strip white spaces
    text = text.translate(remove_punct_map).strip()

    processed_tokens = WordPunctTokenizer().tokenize(text)

    return original_tokens, processed_tokens


def set_device(device) -> torch.device:
    if device in ["cpu", "cuda", "mps"]:
        try:
            device = torch.device(device)
        except Exception as e:
            print(e)
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    return device


def convolve_audio(data: np.ndarray, samplerate: int) -> np.ndarray:
    # create a Hanning kernel 1/50th of a second wide
    kernel_width_seconds = 1.0/50
    kernel_size_points = int(kernel_width_seconds * samplerate)
    kernel = np.hanning(kernel_size_points)

    # normalize the kernel
    kernel = kernel / kernel.sum()

    # Create a filtered signal by convolving the kernel with the original data
    filtered = np.convolve(kernel, data, mode='valid')
    
    return filtered
