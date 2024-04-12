import whisper
import torch

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

class Models():
    def __init__(self, model_task: str="transcribe", model_type: str="pretrained", model_size: str="small", device=None):
        self.model_task = model_task
        self.model_type = model_type
        self.model_size = model_size
        
        self.device = device
    
    def load_vanilla(self):
        model = whisper.load_model(self.model_size)
        
        self.speech_model = model
        
        print("Loaded stock whisper model...")
        
    def load_pretrained(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(f"bofenghuang/whisper-{self.model_size}-cv11-german").to(self.device)
        processor = AutoProcessor.from_pretrained(f"bofenghuang/whisper-{self.model_size}-cv11-german", language="german", task="transcribe")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")
        
        self.speech_model = model
        self.processor = processor
        
        print("Loaded pretrained whisper model...")
    
    def load(self):
        if self.model_task == "tts":
            return None
        elif self.model_type == "pretrained":
            self.load_pretrained()
        elif self.model_type == "vanilla":
            self.load_vanilla()
        else:
            raise ValueError("Model type not supported.")
        
    def check_params(self, model_task, model_type, model_size, device):
        available_model_tasks = ["transcribe", "tts"]
        available_model_types = ["vanilla", "pretrained"]
        available_model_sizes = {"vanilla": ["base", "small", "medium", "large"],
                                 "pretrained": ["small", "medium", "large"]}
        
        
        
        self.model_task = model_task if model_task in available_model_tasks else "transcribe"
        self.model_type = model_type if model_type in available_model_types else "vanilla"
        self.model_size = model_size if model_size in available_model_sizes[self.model_type] else "small"
        self.model_size = "large-v2" if model_size == "large" and self.model_type == "pretrained" else model_size
        
        if model_task not in available_model_tasks:
            print(f"Model task not supported. Defaulting to {self.model_task}.")
            
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
            model_task: {self.model_task}\n\
                model_type: {self.model_type}\n\
                    model_size: {self.model_size}\n\
                        device: {self.device}")
        
    @staticmethod
    def get_models():
        models = {
            "vanilla whisper": ["base", "small", "medium", "large"],
            "pretrained whisper": ["base", "small", "medium", "large-v2"],
        }
        print("Available models:")
        for model_type in models:
            print(f"{model_type}: {models[model_type]}")
    
    @staticmethod
    def set_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        return device