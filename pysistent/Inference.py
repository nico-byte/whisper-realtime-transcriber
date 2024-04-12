import numpy as np
import torch
import asyncio

from .Models import Models
from .utils import preprocess_text


class Inference(Models):
    async def __ainit__(self, model_task: str="transcribe", model_type: str="pretrained", model_size: str="small", device=None):
        super().__init__(model_task, model_type, model_size, device)
        
    async def run_inference(self, audio_data=None):
        if self.model_task == "transcribe" and self.model_type == "vanilla":
            await self.run_vanilla(audio_data)
            
        elif self.model_task == "transcribe" and self.model_type == "pretrained":
            await self.run_pretrained(audio_data)
        
        await self.preprocess()
    
    async def run_vanilla(self, audio_data):
        audio_data_transformed = audio_data.flatten().astype(np.float32) / 32768.0
        # transcribe the time series
        transcript = await asyncio.to_thread(self.speech_model.transcribe, audio_data_transformed, language="de")
        
        self.transcript = transcript['text']
    
    async def run_pretrained(self, audio_data):
        audio_data_transformed = audio_data.flatten().astype(np.float32) / 32768.0
        # Transform the waveform into the required format
        waveform = torch.from_numpy(audio_data_transformed)
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features
        input_features = input_features.to(self.device)
        # Transcribe the audio using the pre-trained model
        generated_ids = await asyncio.to_thread(self.speech_model.generate, inputs=input_features, max_new_tokens=225)
        transcript = await asyncio.to_thread(self.processor.batch_decode, generated_ids, skip_special_tokens=True)

        self.transcript = transcript[0]
        
    async def preprocess(self):
        self.processed_transcript = await asyncio.to_thread(preprocess_text, self.transcript)