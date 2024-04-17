from transcriber.whisper_models.WhisperBase import WhisperBase


class StockWhisper(WhisperBase):
    async def __ainit__(self, inputstream_generator, model_size: str=None, device: str=None):        
        await super().__ainit__(inputstream_generator, device)
        self.available_model_sizes = ["small", "medium", "large-v3"]
        
        self.model_size = model_size if model_size in self.available_model_sizes else "small"
        self.model_size = "large-v3" if model_size == "large" else self.model_size
        
        if model_size not in self.available_model_sizes:
            print(f"Model size not supported. Defaulting to {self.model_size}.")
            
        self.model_id = f"openai/whisper-{self.model_size}"
            
        await self._load()
        
        print(f"Checked model parameters: \n\
            model_id: {self.model_id}\n\
                model_size: {self.model_size}\n\
                    device: {self.device}\n\
                        torch_dtype: {self.torch_dtype}")
        
    async def _load(self):
        await super()._load()
        
        print("Loaded stock whisper model...")
        
    async def run_inference(self):
        await super().run_inference()
        
    async def get_models(self):
        await super().get_models()