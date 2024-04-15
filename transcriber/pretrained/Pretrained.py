import torch
import torchaudio
import numpy as np
import asyncio

from typing import List
from transcriber.utils import tokenize_text, set_device
from async_class import AsyncClass
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class PretrainedWhisper(AsyncClass):
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
        self.speech_model = None
        self.processor = None
        
        self.transcript: str = ""
        self.original_tokens: List = []
        self.processed_tokens: List = []
        
        available_model_sizes = ["small", "medium", "large-v2"]
        
        self.model_size = model_size if model_size in available_model_sizes else "small"
        self.model_size = "large-v2" if model_size == "large" else self.model_size
        
        if model_size not in available_model_sizes:
            print(f"Model size not supported. Defaulting to {self.model_size}.")
        
        self.device = await asyncio.to_thread(set_device, device)
            
        print(f"Checked model parameters: \n\
            model_size: {self.model_size}\n\
                device: {self.device}")
        
    async def load(self):
        """Loads a pre-trained Whisper model for speech recognition.
    
        This method initializes the speech recognition model by loading a pre-trained Whisper model of the specified size. 
        The loaded model and processor are stored in the `speech_model` and `processor` attributes, respectively.
        """
        model = AutoModelForSpeechSeq2Seq.from_pretrained(f"bofenghuang/whisper-{self.model_size}-cv11-german").to(self.device)
        processor = AutoProcessor.from_pretrained(f"bofenghuang/whisper-{self.model_size}-cv11-german", language="german", task="transcribe")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")
        
        self.speech_model = model
        self.processor = processor
        
        print("Loaded pretrained whisper model...")
        
    async def run_inference(self, audio_data: np.array, sample_rate: int):
        """Runs the pretrained speech recognition model on the provided audio data.
    
        Args:
            audio_data (np.array): The audio data to run inference on.
        """
        waveform = torch.from_numpy(audio_data)
        model_sample_rate = self.processor.feature_extractor.sampling_rate
        
        if sample_rate != model_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
            waveform = resampler(waveform)
        
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        
        input_features = inputs.input_features
        input_features = input_features.to(self.device)
        
        generated_ids = await asyncio.to_thread(self.speech_model.generate, inputs=input_features, max_new_tokens=225)
        transcript = await asyncio.to_thread(self.processor.batch_decode, generated_ids, skip_special_tokens=True)

        self.transcript = transcript[0]
        
        self.original_tokens, self.processed_tokens = await asyncio.to_thread(tokenize_text, self.transcript)
        
    @staticmethod
    async def get_models():
        """Prints a dictionary of available Whisper model types and sizes.
    
        The dictionary contains two keys: "vanilla whisper" and "pretrained whisper". Each key maps to a list of available model sizes for that model type.
    
        This function is used to provide information about the Whisper models that can be used in the application.
        """
        models: List = ["small", "medium", "large-v2"]
        print(f"Available models: {models}")