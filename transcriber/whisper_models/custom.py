from utils.decorators import sync_timer
from transcriber.whisper_models.base import WhisperBase


class CustomWhisper(WhisperBase):
    @sync_timer(print_statement="Loaded finetuned whisper model", return_some=False)
    def __init__(self, inputstream_generator, **kwargs):        
        """
        :param inputstream_generator: the generator to use for streaming audio
        :param model_id (str): alternative model id to use for inference
        :param model_size (str): the size of the model to use for inference
        :param language (str): the language to use for tokenizing the model output
        :param device (str): the device to use for inference
        """
        super().__init__(inputstream_generator, **kwargs)
        self.available_model_sizes = ["small", "medium", "large-v2"]
        
        self.model_size = kwargs['model_size']
        self.model_size = "large-v2" if kwargs['model_size'] == "large" else self.model_size
                
        self.model_id = kwargs['model_id'] if kwargs['model_id'] is not None else f"bofenghuang/whisper-{self.model_size}-cv11-german"
        
        self._load()
        
        # Check if generator samplerate matches models samplerate
        if self.inputstream_generator.SAMPLERATE != self.processor.feature_extractor.sampling_rate:
            self.inputstream_generator.SAMPLERATE = self.processor.feature_extractor.sampling_rate
        
        if kwargs['model_size'] not in self.available_model_sizes:
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
        