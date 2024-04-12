import numpy as np
import asyncio

from .Inference import Inference
from .InputStreamGenerator import InputStreamGenerator

class LiveAudioTranscriber(Inference, InputStreamGenerator):
    async def __ainit__(self, model_task: str="transcribe", model_type: str="pretrained", model_size: str="small", device=None):
        super().__init__(model_task, model_type, model_size, device)
        self.check_params(model_task, model_type, model_size, device)
        
        self.transcript = ""
        self.processed_transcript = ""
        
        self.SAMPLERATE = 16000
        self.BLOCKSIZE = 8000
        self.SILENCE_RATIO = 2000
        self.ADJUSTMENT_TIME = 5
        self.SILENCE_THRESHOLD = 1500
        
        self.global_ndarray = None
        self.temp_ndarray = None
        
    async def transcribe(self):
        """
        Processes audio input streams to detect voice activity and manage buffer concatenation.
        This asynchronous method iterates over generated audio chunks, applying a silence detection
        algorithm to filter out mostly silent buffers. It concatenates non-silent buffers into a global
        array, handling them differently based on the presence of a wake word or the activation of
        automatic speech recognition (ASR). The method dynamically adjusts its behavior to optimize
        for either wake word detection or ASR processing, ensuring efficient memory usage and
        real-time processing capabilities.
        """
        print("Generating transcript...")
        
        async for indata, _ in self.generate():
            indata_flattened = abs(indata.flatten())
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
                self.global_ndarray = None
        
        del self.global_ndarray
        del self.temp_ndarray
        
    async def print_transcriptions(self):
        char_limit = 144  # The character limit after which a new line should start
        current_line_length = 0  # Current length of the line being printed

        # Calculate the new line length if the update is added
        new_line_length = current_line_length + len(self.transcript)
        if new_line_length > char_limit:
            print()  # Start a new line if the limit is exceeded
            current_line_length = 0  # Reset the current line length
        print(self.transcript + " ", end='', flush=True)  # Print the update without a newline, flush to ensure it's displayed
        current_line_length += len(self.transcript) # Update the current line length