from utils.decorators import sync_timer
from transcriber.whisper_models.base import WhisperBase


class DistilWhisper(WhisperBase):
    @sync_timer(print_statement="Loaded distilled whisper model", return_some=False)
    def __init__(self, inputstream_generator, model_size='small', punctuate_truecase=False, device='cpu'):        
        """
        :param inputstream_generator: the generator to use for streaming audio
        :param model_size (str): the size of the model to use for inference
        :param language (str): the language to use for tokenizing the model output
        :param device (str): the device to use for inference
        """
        super().__init__(inputstream_generator, punctuate_truecase, device)
        self.available_model_sizes = ["small", "medium", "large-v3"]
        
        self.model_size = model_size
        self.model_size = "large-v3" if model_size == "large" else self.model_size
                
        self.model_id = f"distil-whisper/distil-{self.model_size}.en" if self.model_size in self.available_model_sizes[:2] else f"distil-whisper/distil-{self.model_size}"
            
        self._load()
        
        # Check if generator samplerate matches models samplerate
        if self.inputstream_generator.SAMPLERATE != self.processor.feature_extractor.sampling_rate:
            self.inputstream_generator.SAMPLERATE = self.processor.feature_extractor.sampling_rate
        
        if model_size not in self.available_model_sizes:
            print(f"Model size not supported. Defaulting to {self.model_size}.")
        
        print(f"Checked model parameters: \n\
            model_id: {self.model_id}\n\
                device: {self.device}\n\
                    torch_dtype: {self.torch_dtype}")
        
    def _load(self):
        super()._load()
                
    async def run_inference(self):
        await super().run_inference()
        
    def get_models(self):
        super().get_models()