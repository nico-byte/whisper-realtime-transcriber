import numpy as np
import asyncio
import sounddevice as sd
import noisereduce as nr

from typing import Tuple, List
from async_class import AsyncClass


class LiveAudioTranscriber(AsyncClass):
    async def __ainit__(self, samplerate: int=None, blocksize: int=None, adjustment_time: int=None, silence_threshold: float=None):
        self.SAMPLERATE = 16000 if samplerate is None else samplerate
        self.BLOCKSIZE = 8000 if blocksize is None else blocksize
        self.ADJUSTMENT_TIME = 5 if adjustment_time is None else adjustment_time
        
        self.SILENCE_THRESHOLD = silence_threshold
        
        if self.SILENCE_THRESHOLD is None:
            await self.set_silence_threshold()
        
        self.global_ndarray: np.ndarray = None
        self.temp_ndarray: np.ndarray = None
                
        print(f"Checked inputstream parameters: \n\
            samplerate: {self.SAMPLERATE}\n\
                blocksize: {self.BLOCKSIZE}")
    async def generate(self):
        q_in = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(in_data, _, __, state):
            loop.call_soon_threadsafe(q_in.put_nowait, (in_data.copy(), state))

        stream = sd.InputStream(samplerate=self.SAMPLERATE, channels=1, dtype='int16', blocksize=self.BLOCKSIZE, callback=callback)
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata, status
    
    async def transcribe(self, model, loop_forever: bool) -> Tuple[List[str], List[str], List[str]]:
        print("Listening...")
        
        async for indata, _ in self.generate():
            indata_flattened: np.ndarray = abs(indata.flatten())
            
            # discard buffers that contain mostly silence
            if (np.percentile(indata_flattened, 10) <= self.SILENCE_THRESHOLD) and self.global_ndarray is None:
                continue
            if self.global_ndarray is not None:
                self.global_ndarray = np.concatenate((self.global_ndarray, indata), dtype='int16')
            else:
                self.global_ndarray = indata
            # concatenate buffers if the end of the current buffer is not silent
            if (np.percentile(indata_flattened[-100:-1], 10) > self.SILENCE_THRESHOLD):
                continue
            else:
                self.temp_ndarray = self.global_ndarray.copy()
                self.temp_ndarray = self.temp_ndarray.flatten().astype(np.float32) / 32768.0
                self.temp_ndarray = await asyncio.to_thread(nr.reduce_noise, y=self.temp_ndarray, sr=self.SAMPLERATE)

                await model.run_inference(self.temp_ndarray, self.SAMPLERATE)
                
                if loop_forever:
                    await self.print_transcriptions(model.transcript)
                    self.global_ndarray: np.ndarray = None
                    self.temp_ndarray: np.ndarray = None
                else:
                    return model.transcript, model.original_tokens, model.processed_tokens
        
    @staticmethod
    async def print_transcriptions(transcript):
        char_limit: int = 77  # The character limit after which a new line should start
        current_line_length: int = 0  # Current length of the line being printed

        # Calculate the new line length if the update is added
        new_line_length: int = current_line_length + len(transcript)
        if new_line_length > char_limit:
            print()  # Start a new line if the limit is exceeded
            current_line_length = 0  # Reset the current line length
        print(transcript + " ", end='', flush=True)  # Print the update without a newline, flush to ensure it's displayed
        current_line_length += len(transcript) # Update the current line length
        
    async def set_silence_threshold(self):
        blocks_processed: int = 0
        loudness_values: list = []

        async for indata, _ in self.generate():
            blocks_processed += 1
            indata_flattened: np.ndarray = abs(indata.flatten())

            # Compute loudness over first few seconds to adjust silence threshold
            loudness_values.append(np.mean(indata_flattened))

            # Stop recording after ADJUSTMENT_TIME seconds
            if blocks_processed >= self.ADJUSTMENT_TIME * (self.SAMPLERATE / self.BLOCKSIZE):
                self.SILENCE_THRESHOLD = float(np.percentile(loudness_values, 50))
                break
            
        print(f'\nSet SILENCE_THRESHOLD to {self.SILENCE_THRESHOLD}\n')
