import whisper
import torch
import numpy as np
import asyncio

from typing import List
from transcriber.utils import tokenize_text, set_device
from async_class import AsyncClass


class StockWhisper(AsyncClass):
    """
    Provides a class for loading and managing Whisper speech recognition models.
    ----------------------------------------------------------------------------
    Parameters
    ----------
    model_size: str
        The size of the model to use. Default: "small"
    device: str
        The device to use for PyTorch operations. Default: "cpu" if cuda or mps not available
    """
    async def __ainit__(self, model_size: str=None, device: str=None):
        await self.check_params(model_size, device)
        
        self.speech_model = None
        
        self.transcript: str = ""
        self.original_tokens: List = []
        self.processed_tokens: List = []
        
        available_model_sizes = ["base", "small", "medium", "large"]
        
        self.model_size = model_size if model_size in available_model_sizes else "small"
        
        if model_size not in available_model_sizes:
            print(f"Model size not supported. Defaulting to {self.model_size}.")
        
        self.device = await asyncio.to_thread(set_device, device)
            
        print(f"Checked model parameters: \n\
            model_size: {self.model_size}\n\
                device: {self.device}")
    
    async def load(self):
        """Loads the stock Whisper model of the specified size.
    
        This method initializes the speech recognition model by loading the pre-trained Whisper model of the specified size. 
        The loaded model is stored in the `speech_model` attribute.
        """
        model = whisper.load_model(self.model_size)
        
        self.speech_model = model
        
        print("Loaded stock whisper model...")
        
    async def run_inference(self, audio_data: np.array):
        """Runs the vanilla speech recognition model on the provided audio data.
    
        Args:
            audio_data (np.array): The audio data to run inference on.
        """
        transcript = await asyncio.to_thread(self.speech_model.transcribe, audio_data, language="de")
        
        self.transcript = transcript['text']
        
        self.original_tokens, self.processed_tokens = await asyncio.to_thread(tokenize_text, self.transcript)
        
    @staticmethod
    async def get_models():
        """Prints a dictionary of available Whisper model types and sizes.
    
        The dictionary contains two keys: "vanilla whisper" and "pretrained whisper". Each key maps to a list of available model sizes for that model type.
    
        This function is used to provide information about the Whisper models that can be used in the application.
        """
        models: List = ["base", "small", "medium", "large"]
        print("Available models:")
        print(f"Available models: {models}")