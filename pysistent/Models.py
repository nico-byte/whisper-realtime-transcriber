import whisper
import torch

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class Models():
    """
    Provides a class for loading and managing Whisper speech recognition models.
    ----------------------------------------------------------------------------
    Parameters
    ----------
    model_tpye: str
        The model_type to use. Default: "pretrained"
    model_size: str
        The size of the model to use. Default: "small"
    device: str
        The device to use for PyTorch operations. Default: None
    """
    def __init__(self, model_type: str="pretrained", model_size: str="small", device: str=None):
        self.model_type = model_type
        self.model_size = model_size
        
        self.device = device
    
    def load_vanilla(self):
        """Loads the stock Whisper model of the specified size.
    
        This method initializes the speech recognition model by loading the pre-trained Whisper model of the specified size. 
        The loaded model is stored in the `speech_model` attribute.
        """
        model = whisper.load_model(self.model_size)
        
        self.speech_model = model
        
        print("Loaded stock whisper model...")
        
    def load_pretrained(self):
        """Loads a pre-trained Whisper model for speech recognition.
    
        This method initializes the speech recognition model by loading a pre-trained Whisper model of the specified size. 
        The loaded model and processor are stored in the `speech_model` and `processor` attributes, respectively.
        """
        model = AutoModelForSpeechSeq2Seq.from_pretrained(f"bofenghuang/whisper-{self.model_size}-cv11-german").to(self.device)
        processor = AutoProcessor.from_pretrained(f"bofenghuang/whisper-{self.model_size}-cv11-german", language="german", task="transcribe")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")
        
        self.speech_model = model
        self.processor = processor
        
        print("Loaded pretrained whisper model...")
    
    def load(self):
        """Loads the appropriate Whisper model based on the specified model type.
    
        This method checks the `model_type` attribute and loads either the stock Whisper model or a pre-trained Whisper model. 
        If the `model_type` is "pretrained", it loads the pre-trained model using the `load_pretrained()` method. 
        If the `model_type` is "vanilla", it loads the stock Whisper model using the `load_vanilla()` method. 
        If the `model_type` is neither "pretrained" nor "vanilla", it raises a `ValueError`.
        
        Raises:
            ValueError: If the provided `model_type` is not supported.
        """
        if self.model_type == "pretrained":
            self.load_pretrained()
        elif self.model_type == "vanilla":
            self.load_vanilla()
        else:
            raise ValueError("Model type not supported.")
        
    def check_params(self, model_type: str, model_size: str, device: str=None):
        """Checks the validity of the provided model parameters and sets the appropriate values.
    
        This method checks the provided `model_type` and `model_size` parameters to ensure they are valid. 
        If the provided values are not valid, it sets the parameters to the default values. 
        It also sets the `device` parameter based on the provided value, falling back to CPU if the provided device is not available.
    
        The method prints out the final values of the model parameters for debugging purposes.
    
        Args:
            model_type (str): The type of the Whisper model to use. Can be either "vanilla" or "pretrained".
            model_size (str): The size of the Whisper model to use. The available sizes depend on the `model_type`.
            device (str, optional): The device to use for the Whisper model. Can be "cpu", "cuda", or "mps". If not provided, the method will attempt to set the device automatically.
        """
        available_model_types = ["vanilla", "pretrained"]
        available_model_sizes = {"vanilla": ["base", "small", "medium", "large"],
                                 "pretrained": ["small", "medium", "large"]}
        
        
        
        self.model_type = model_type if model_type in available_model_types else "vanilla"
        self.model_size = model_size if model_size in available_model_sizes[self.model_type] else "small"
        self.model_size = "large-v2" if model_size == "large" and self.model_type == "pretrained" else model_size
            
        if model_type not in available_model_types:
            print(f"Model type not supported. Defaulting to {self.model_type}.")
        
        if model_size not in available_model_sizes[self.model_type]:
            print(f"Model size not supported. Defaulting to {self.model_size}.")
        
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
            
        print(f"Checked model parameters: \n\
            model_type: {self.model_type}\n\
                model_size: {self.model_size}\n\
                    device: {self.device}")
        
    @staticmethod
    def get_models():
        """Prints a dictionary of available Whisper model types and sizes.
    
        The dictionary contains two keys: "vanilla whisper" and "pretrained whisper". Each key maps to a list of available model sizes for that model type.
    
        This function is used to provide information about the Whisper models that can be used in the application.
        """
        models: dict = {
            "vanilla whisper": ["base", "small", "medium", "large"],
            "pretrained whisper": ["base", "small", "medium", "large-v2"],
        }
        print("Available models:")
        for model_type in models:
            print(f"{model_type}: {models[model_type]}")
    
    @staticmethod
    def set_device() -> torch.device:
        """Determines the appropriate device to use for PyTorch operations, prioritizing GPU and MPS (Apple Silicon) devices if available, and falling back to CPU if neither is available.
    
        Returns:
            torch.device: The device to use for PyTorch operations.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        return device