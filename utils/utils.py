import re
import string
import torch

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
