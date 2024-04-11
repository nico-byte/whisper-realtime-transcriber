import whisper
import torch
import asyncio

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from speechbrain.inference import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from async_class import AsyncClass

class Model(AsyncClass):
    async def __ainit__(self, model_task="transcribe", model_type="vanilla", model_size="base", device=None):
        self.model_task = model_task
        self.model_type = model_type
        self.model_size = model_size
        
        self.device = device
    
    async def load_vanilla(self):
        model = whisper.load_model(self.model_size)
        
        self.speech_model = model
        
    async def load_pretrained(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(f"bofenghuang/whisper-{self.model_size}-cv11-german").to(self.device)
        processor = AutoProcessor.from_pretrained(f"bofenghuang/whisper-{self.model_size}-cv11-german", language="german", task="transcribe")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")
        
        self.speech_model = model
        self.processor = processor
    
    async def load_tts(self):
        tacotron2 = Tacotron2.from_hparams(source="padmalcom/tts-tacotron2-german", savedir="tmpdir_tts")
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
        
        self.speech_model = tacotron2
        self.processor = hifi_gan
    
    async def load(self):
        if self.model_task == "tts":
            await self.load_tts()
        elif self.model_type == "pretrained":
            await self.load_pretrained()
        elif self.model_type == "vanilla":
            await self.load_vanilla()
        else:
            raise ValueError("Model type not supported.")
        
    @staticmethod
    async def set_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        return device