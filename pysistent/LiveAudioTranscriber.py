import numpy as np
import asyncio

from .Inference import Inference
from .InputStreamGenerator import InputStreamGenerator

class LiveAudioTranscriber(Inference, InputStreamGenerator):
    """
    Provides a class for combining the Inference class and InputStreamGenerator class.
    ----------------------------------------------------------------------------------
    Paramters
    ----------
    model_tpye: str
        The model_type to use. Default: "pretrained"
    model_size: str
        The size of the model to use. Default: "small"
    device: str
        The device to use for PyTorch operations. Default: None
    samplerate: int
        The samplerate to use for the input stream. Default: 16000
    blocksize: int
        The size of the blocks to use for the input stream. Default: 8000
    silence_ratio: int
        The max amount of silent values in one block. Default: 3000
    ajustment_time: int
        The duration used for generating the silence_threshold. Default: 5
    silence_threshold: int
        If it is not desired to auto generate a silence_threshold, set this value to a desired value. Default: 1500/15
    """
    async def __ainit__(self, model_type: str="pretrained", model_size: str="small", device: str=None, 
                        samplerate: int=16000, blocksize: int=8000, silence_ratio: int=3000, adjustment_time: int=5, silence_threshold: int=1500/15):
        super().__init__(model_type, model_size, device)
        self.check_params(model_type, model_size, device)
        
        self.transcript: str = ""
        self.processed_transcript: str = ""
        
        self.SAMPLERATE = samplerate
        self.BLOCKSIZE = blocksize
        self.SILENCE_RATIO = silence_ratio
        self.ADJUSTMENT_TIME = adjustment_time
        self.SILENCE_THRESHOLD = silence_threshold
        
        self.global_ndarray: np.ndarray = None
        self.temp_ndarray: np.ndarray = None
        
    async def transcribe(self):
        """Transcribes live audio data by continuously processing audio buffers and generating a transcript.

        This method is responsible for the following:
        - Generating audio data using the `generate()` method.
        - Filtering out audio buffers that contain mostly silence.
        - Concatenating non-silent audio buffers into a larger buffer.
        - Running inference on the accumulated audio buffer and printing the transcription.
        - Resetting the accumulated audio buffer when it exceeds a certain size.
        """
        print("Generating transcript...")
        
        async for indata, _ in self.generate():
            indata_flattened: np.ndarray = abs(indata.flatten())
            # discard buffers that contain mostly silence
            if len(np.nonzero(indata_flattened > self.SILENCE_THRESHOLD)[0]) < self.SILENCE_RATIO:
                continue
            if self.global_ndarray is not None:
                self.global_ndarray = np.concatenate((self.global_ndarray, indata), dtype='int16')
            else:
                self.global_ndarray = indata
            # concatenate buffers if the end of the current buffer is not silent and if the chunksize is under 5
            if (np.average((indata_flattened[-100:-1])) > self.SILENCE_THRESHOLD):
                continue
            if (len(self.global_ndarray) > self.BLOCKSIZE * 3):
                self.temp_ndarray = self.global_ndarray.copy()
                await self.run_inference(self.temp_ndarray)
                await self.print_transcriptions()
                self.global_ndarray: np.ndarray = None
        
        del self.global_ndarray
        del self.temp_ndarray
        
    async def print_transcriptions(self):
        """Prints the current transcript, ensuring that the output is formatted with a maximum line length of 144 characters. 
        
        If the addition of the current transcript would exceed the line length, a new line is started.
        """
        char_limit: int = 144  # The character limit after which a new line should start
        current_line_length: int = 0  # Current length of the line being printed

        # Calculate the new line length if the update is added
        new_line_length: int = current_line_length + len(self.transcript)
        if new_line_length > char_limit:
            print()  # Start a new line if the limit is exceeded
            current_line_length = 0  # Reset the current line length
        print(self.transcript + " ", end='', flush=True)  # Print the update without a newline, flush to ensure it's displayed
        current_line_length += len(self.transcript) # Update the current line length