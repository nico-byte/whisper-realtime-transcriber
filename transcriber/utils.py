import re
import string
import torch

from typing import Tuple, List
from num2words import num2words
from nltk.tokenize import WordPunctTokenizer


def tokenize_text(text: str) -> Tuple[List[str], List[str]]:
    """Tokenizes the given text and performs various transformations, including:
    
    - Keeping the original tokens
    - Converting the text to lowercase
    - Identifying and converting times to words
    - Converting numbers to words
    - Removing punctuation while preserving German umlauts
    - Stripping whitespace

    Args:
        text (str): The text to be preprocessed.

    Returns:
        Tuple[List[str], List[str]]: The original and processed tokens.
    """
    # Keep original tokens
    original_tokens =  WordPunctTokenizer().tokenize(text)
    
    # Convert text to lower case
    text = text.lower()
    # Identify and convert times
    time_pattern = r"\b\d{1,2}:\d{2}\b"
    times = re.findall(time_pattern, text)
    for time in times:
        hours, minutes = map(int, time.split(":"))
        time_in_words = f"{num2words(hours, lang='de')} Uhr {num2words(minutes, lang='de')}"
        text = text.replace(time, time_in_words)

    # Convert numbers to words
    # ordinal numbers will be converted to usual numbers - 2nd will be two
    text = ' '.join(num2words(int(word), lang='de', ordinal=False) if word.isdigit() else word for word in text.split())
    
    # Keep German umlauts
    remove_punct_map = {ord(char): None for char in string.punctuation if char not in ['ä', 'ö', 'ü', 'ß']}
    
    # Remove punctuation and strip white spaces
    text = text.translate(remove_punct_map).strip()

    processed_tokens = WordPunctTokenizer().tokenize(text)

    return original_tokens, processed_tokens


def set_device(device) -> torch.device:
    """Determines the appropriate device to use for PyTorch operations, prioritizing GPU and MPS (Apple Silicon) devices if available, and falling back to CPU if neither is available.

    Returns:
        torch.device: The device to use for PyTorch operations.
    """
    if device in ["cpu", "cuda", "mps"]:
        try:
            device = torch.device(device)
        except Exception as e:
            print(e)
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    return device
