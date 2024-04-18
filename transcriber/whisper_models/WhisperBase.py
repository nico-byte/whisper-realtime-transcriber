import torch
import asyncio

from typing import List
from utils.utils import tokenize_text, set_device
from utils.decorators import async_timer
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class WhisperBase():
    def __init__(self, inputstream_generator, language: str=None, device: str=None):        
        self.speech_model = None
        self.processor = None
        
        self.inputstream_generator = inputstream_generator
        
        self.language = "en" if language is None else language
                
        self.transcript: str = ""
        self.original_tokens: List = []
        self.processed_tokens: List = []
        
        self.gen_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 1,
            "return_timestamps": False,
            }
                    
        self.device = set_device(device)
        
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.inputstream_generator = inputstream_generator
        
    def _load(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.speech_model = model
        self.processor = processor
                
    async def run_inference(self):
        while True:
            await self.inputstream_generator.data_ready_event.wait()
            
            transcription_duration = await self._transcribe()
                        
            audio_duration = len(self.inputstream_generator.temp_ndarray) / self.inputstream_generator.SAMPLERATE
            realtime_factor = transcription_duration / audio_duration
            
            if realtime_factor > 1:
                print(f"\nTranscription took longer ({transcription_duration:.3f}s) than length of input in seconds ({audio_duration:.3f}s).")
                print(f"Real-Time Factor: {realtime_factor:.3f}, try to use a smaller model.")
                print("Exiting now, to avoid potential memory issues...")
                raise asyncio.CancelledError()
            
            await self._print_transcriptions()
            
            self.inputstream_generator.data_ready_event.clear()
            
    @async_timer
    async def _transcribe(self):
        waveform = torch.from_numpy(self.inputstream_generator.temp_ndarray)

        input_features = self.processor(waveform, sampling_rate=self.inputstream_generator.SAMPLERATE, return_tensors="pt").input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)
        generated_ids = await asyncio.to_thread(self.speech_model.generate, input_features=input_features, **self.gen_kwargs)
        transcript = await asyncio.to_thread(self.processor.batch_decode, generated_ids, skip_special_tokens=True, decode_with_timestamps=self.gen_kwargs["return_timestamps"])
        self.transcript = transcript[0]
        self.original_tokens, _ = await asyncio.to_thread(tokenize_text, self.transcript, self.language)
        
    async def _print_transcriptions(self):
        char_limit: int = 77  # The character limit after which a new line should start
        current_line_length: int = 0  # Current length of the line being printed

        # Calculate the new line length if the update is added
        new_line_length: int = current_line_length + len(self.transcript)
        if new_line_length > char_limit:
            print()  # Start a new line if the limit is exceeded
            current_line_length = 0  # Reset the current line length
        print(self.transcript + " ", end='', flush=True)  # Print the update without a newline, flush to ensure it's displayed
        current_line_length += len(self.transcript) # Update the current line length
    
    def get_models(self):
        print(f"Available models: {self.available_model_sizes}")