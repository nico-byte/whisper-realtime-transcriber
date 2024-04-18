from utils.decorators import init_timer
from transcriber.whisper_models.WhisperBase import WhisperBase


class StockWhisper(WhisperBase):
    @init_timer(print_statement="Loaded stock whisper model")
    def __init__(self, inputstream_generator, model_size: str=None, language: str=None, device: str=None):        
        super().__init__(inputstream_generator, language, device)
        self.available_model_sizes = ["small", "medium", "large-v3"]
                
        self.model_size = model_size if model_size in self.available_model_sizes else "small"
        self.model_size = "large-v3" if model_size == "large" else self.model_size
                
        self.model_id = f"openai/whisper-{self.model_size}"
            
        self._load()
        
        if self.inputstream_generator.SAMPLERATE != self.processor.feature_extractor.sampling_rate:
            self.inputstream_generator.SAMPLERATE = self.processor.feature_extractor.sampling_rate
        
        if model_size not in self.available_model_sizes:
            print(f"Model size not supported. Defaulting to {self.model_size}.")
        
        print(f"Checked model parameters: \n\
            model_id: {self.model_id}\n\
                model_size: {self.model_size}\n\
                    device: {self.device}\n\
                        torch_dtype: {self.torch_dtype}")
        
    def _load(self):
        super()._load()
                
    async def run_inference(self):
        await super().run_inference()
        
    def get_models(self):
        super().get_models()