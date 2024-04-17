import numpy as np
import asyncio
import sounddevice as sd

from utils.decorators import async_timer
from async_class import AsyncClass


class InputStreamGenerator(AsyncClass):
    @async_timer(print_value=True, statement="Loaded inputstream generator")
    async def __ainit__(self, samplerate: int=None, blocksize: int=None, adjustment_time: int=None, silence_threshold: float=None):
        self.SAMPLERATE = 16000 if samplerate is None else samplerate
        self.BLOCKSIZE = 4000 if blocksize is None else blocksize
        self.ADJUSTMENT_TIME = 5 if adjustment_time is None else adjustment_time
        
        self.SILENCE_THRESHOLD = silence_threshold
        
        if self.SILENCE_THRESHOLD is None:
            await self._set_silence_threshold()
        
        self.global_ndarray: np.ndarray = None
        self.temp_ndarray: np.ndarray = None
        
        self.data_ready_event = asyncio.Event()
                
        print(f"Checked inputstream parameters: \n\
            samplerate: {self.SAMPLERATE}\n\
                blocksize: {self.BLOCKSIZE}")
    
    async def _generate(self):
        q_in = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(in_data, _, __, state):
            loop.call_soon_threadsafe(q_in.put_nowait, (in_data.copy(), state))

        stream = sd.InputStream(samplerate=self.SAMPLERATE, channels=1, dtype='int16', blocksize=self.BLOCKSIZE, callback=callback)
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata, status
    
    async def process_audio(self):
        print("Listening...")
        
        async for indata, _ in self._generate():
            indata_flattened: np.ndarray = abs(indata.flatten())
            
            # discard buffers that contain mostly silence
            if (np.percentile(indata_flattened, 10) <= self.SILENCE_THRESHOLD) and self.global_ndarray is None:
                continue
            if self.global_ndarray is not None:
                self.global_ndarray = np.concatenate((self.global_ndarray, indata), dtype='int16')
            else:
                self.global_ndarray = indata
            # concatenate buffers if the end of the current buffer is not silent
            if (np.percentile(indata_flattened[-100:-1], 10) > self.SILENCE_THRESHOLD) or self.data_ready_event.is_set():
                continue
            else:
                self.temp_ndarray = self.global_ndarray.copy()
                self.temp_ndarray = self.temp_ndarray.flatten().astype(np.float32) / 32768.0
                
                self.global_ndarray = None
                self.data_ready_event.set()
        
    async def _set_silence_threshold(self):
        blocks_processed: int = 0
        loudness_values: list = []

        async for indata, _ in self._generate():
            blocks_processed += 1
            indata_flattened: np.ndarray = abs(indata.flatten())

            # Compute loudness over first few seconds to adjust silence threshold
            loudness_values.append(np.mean(indata_flattened))

            # Stop recording after ADJUSTMENT_TIME seconds
            if blocks_processed >= self.ADJUSTMENT_TIME * (self.SAMPLERATE / self.BLOCKSIZE):
                self.SILENCE_THRESHOLD = float(np.percentile(loudness_values, 50))
                break
            
        print(f'\nSet SILENCE_THRESHOLD to {self.SILENCE_THRESHOLD}\n')
