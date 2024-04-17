import torch
import torchaudio
import asyncio
import time

from typing import List
from transcriber.utils import tokenize_text, set_device
from async_class import AsyncClass
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class WhisperBase(AsyncClass):
    async def __ainit__(self, inputstream_generator, language: str=None, device: str=None):        
        self.speech_model = None
        self.processor = None
        
        self.language = language if language is not None else "en"
        
        self.transcript: str = ""
        self.original_tokens: List = []
        self.processed_tokens: List = []
                    
        self.device = await asyncio.to_thread(set_device, device)
        
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.inputstream_generator = inputstream_generator
        
    async def load(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id, language=self.language, task="transcribe")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
        
        self.speech_model = model
        self.processor = processor
                
    async def run_inference(self):
        gen_kwargs = {
              "max_new_tokens": 128,
              "num_beams": 1,
              "return_timestamps": False,
            }
        
        while True:
            await self.inputstream_generator.data_ready_event.wait()
            start_time = time.monotonic()
            waveform = torch.from_numpy(self.inputstream_generator.temp_ndarray)
            model_sample_rate = self.processor.feature_extractor.sampling_rate

            if self.inputstream_generator.SAMPLERATE != model_sample_rate:
                resampler = torchaudio.transforms.Resample(self.inputstream_generator.SAMPLERATE, model_sample_rate)
                waveform = resampler(waveform)

            input_features = self.processor(waveform, sampling_rate=self.inputstream_generator.SAMPLERATE, return_tensors="pt").input_features

            input_features = input_features.to(self.device, dtype=self.torch_dtype)


            generated_ids = await asyncio.to_thread(self.speech_model.generate, inputs=input_features, **gen_kwargs)
            transcript = await asyncio.to_thread(self.processor.batch_decode, generated_ids, skip_special_tokens=True, decode_with_timestamps=gen_kwargs["return_timestamps"])

            self.transcript = transcript[0]

            self.original_tokens, self.processed_tokens = await asyncio.to_thread(tokenize_text, self.transcript)
            
            await self.print_transcriptions()
            
            end_time = time.monotonic()
            
            realtime_factor = (end_time - start_time) / (len(self.inputstream_generator.temp_ndarray) / self.inputstream_generator.SAMPLERATE)
            
            if realtime_factor > 1:
                raise RuntimeError("Transcription took longer than length of input in seconds, try to use a smaller model.")
            
            self.inputstream_generator.data_ready_event.clear()
        
    async def print_transcriptions(self):
        char_limit: int = 77  # The character limit after which a new line should start
        current_line_length: int = 0  # Current length of the line being printed

        # Calculate the new line length if the update is added
        new_line_length: int = current_line_length + len(self.transcript)
        if new_line_length > char_limit:
            print()  # Start a new line if the limit is exceeded
            current_line_length = 0  # Reset the current line length
        print(self.transcript + " ", end='', flush=True)  # Print the update without a newline, flush to ensure it's displayed
        current_line_length += len(self.transcript) # Update the current line length
    
    async def get_models(self):
        print(f"Available models: {self.available_models}")