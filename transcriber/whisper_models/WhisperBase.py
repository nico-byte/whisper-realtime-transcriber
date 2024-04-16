import torch
import torchaudio
import numpy as np
import asyncio

from typing import List
from transcriber.utils import tokenize_text, set_device
from async_class import AsyncClass
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class WhisperBase(AsyncClass):
    async def __ainit__(self, language: str=None, device: str=None):        
        self.speech_model = None
        self.processor = None
        
        self.language = language if language is not None else "en"
        
        self.transcript: str = ""
        self.original_tokens: List = []
        self.processed_tokens: List = []
                    
        self.device = await asyncio.to_thread(set_device, device)
        
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
    async def load(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id, language=self.language, task="transcribe")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
        
        self.speech_model = model
        self.processor = processor
                
    async def run_inference(self, audio_data: np.array, sample_rate: int):
        waveform = torch.from_numpy(audio_data)
        model_sample_rate = self.processor.feature_extractor.sampling_rate
        
        if sample_rate != model_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
            waveform = resampler(waveform)
        
        input_features = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features
        
        input_features = input_features.to(self.device, dtype=self.torch_dtype)
        
        gen_kwargs = {
          "max_new_tokens": 128,
          "num_beams": 1,
          "return_timestamps": False,
        }

        
        generated_ids = await asyncio.to_thread(self.speech_model.generate, inputs=input_features, **gen_kwargs)
        transcript = await asyncio.to_thread(self.processor.batch_decode, generated_ids, skip_special_tokens=True, decode_with_timestamps=gen_kwargs["return_timestamps"])

        self.transcript = transcript[0]
        
        self.original_tokens, self.processed_tokens = await asyncio.to_thread(tokenize_text, self.transcript)
        
    async def get_models(self):
        print(f"Available models: {self.available_models}")