import numpy as np
import asyncio
import torch
import torchaudio

from Model import Model
from text_processing import preprocess_text
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav


class Inference(Model):
    def __init__(self, model_task="transcribe", model_type="vanilla", model_size="base", device=None):
        super(Inference, self).__init__(model_task, model_type, model_size, device)
        available_model_tasks = ["transcribe", "tts"]
        available_model_types = ["vanilla", "pretrained"]
        available_model_sizes = ["base", "small", "medium", "large"]
        
        self.model_task = model_task if model_task in available_model_tasks else "transcribe"
        self.model_type = model_type if model_type in available_model_types else "vanilla"
        self.model_size = model_size if model_size in available_model_sizes else "base"
        self.model_size = "large-v2" if model_size == "large" and self.model_type == "pretrained" else model_size
        self.model_size = "small" if model_size == "base" and self.model_type == "pretrained" else model_size
        
        if device is None:
            self.device = self.set_device()
        elif device in ["cpu", "cuda", "mps"]:
            try:
                self.device = torch.device(device)
            except Exception as e:
                print(e)
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        self.transcript = ""
        self.processed_transcript = ""
        
        self.path_to_file = 'tts_output.wav'

    def run(self, audio_data=None, text=None):
        if self.model_task == "tts":
            self.run_tts(text)
        
        elif self.model_task == "transcribe" and self.model_type == "vanilla":
            self.run_vanilla(audio_data)
            
        elif self.model_task == "transcribe" and self.model_type == "pretrained":
            self.run_pretrained(audio_data)
        
        self.preprocess()
            
    def run_tts(self, text):
        # Running the TTS
        audio_array = generate_audio(text)
        write_wav(self.path_to_file, SAMPLE_RATE, audio_array)
    
    def run_vanilla(self, audio_data):
        audio_data_transformed = audio_data.flatten().astype(np.float32) / 32768.0
        # transcribe the time series
        transcript = self.speech_model.transcribe(audio_data_transformed, language="de")
        
        self.transcript = transcript['text']
    
    def run_pretrained(self, audio_data):
        audio_data_transformed = audio_data.flatten().astype(np.float32) / 32768.0
        # Transform the waveform into the required format
        waveform = torch.from_numpy(audio_data_transformed)
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features
        input_features = input_features.to(self.device)
        # Transcribe the audio using the pre-trained model
        generated_ids = self.speech_model.generate(inputs=input_features, max_new_tokens=225)
        transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.transcript = transcript
        
    def preprocess(self):
        self.processed_transcript = preprocess_text(self.transcript)