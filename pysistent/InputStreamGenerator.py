import numpy as np
import sounddevice as sd
import asyncio
import warnings
import time

from async_class import AsyncClass


class InputStreamGenerator(AsyncClass):
    """
    Represents a generator that produces an input stream of audio data.
    """
    async def __ainit__(self, samplerate, blocksize, silence_ratio, adjustment_time):
        self.SAMPLERATE = samplerate
        self.BLOCKSIZE = blocksize
        self.SILENCE_THRESHOLD = None
        self.SILENCE_RATIO = silence_ratio
        self.ADJUSTMENT_TIME = adjustment_time
        
        self.global_ndarray = None
        self.temp_ndarray = None

    async def generate(self):
        """
        This asynchronous generator function initiates an audio input stream with the specified sample rate and block size,
        using sounddevice. It continuously reads audio data into a queue in a non-blocking manner and yields the data along with its status
        whenever available. This function is designed to be used in an asynchronous context to facilitate real-time audio processing tasks.
        """
        q_in = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(in_data, _, __, state):
            loop.call_soon_threadsafe(q_in.put_nowait, (in_data.copy(), state))

        stream = sd.InputStream(samplerate=self.SAMPLERATE, channels=1, dtype='int16', blocksize=self.BLOCKSIZE, callback=callback)
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata, status
                
    async def record(self, duration):
        """
        Processes audio input streams to detect voice activity and manage buffer concatenation.
    
        This asynchronous method iterates over generated audio chunks, applying a silence detection
        algorithm to filter out mostly silent buffers. It concatenates non-silent buffers into a global
        array, handling them differently based on the presence of a wake word or the activation of
        automatic speech recognition (ASR). The method dynamically adjusts its behavior to optimize
        for either wake word detection or ASR processing, ensuring efficient memory usage and
        real-time processing capabilities.
        """
        start_time = time.monotonic()
        async for indata, _ in self.generate():
            indata_flattened = abs(indata.flatten())

            # discard buffers that contain mostly silence
            if len(np.nonzero(indata_flattened > self.SILENCE_THRESHOLD)[0]) < self.SILENCE_RATIO:
                continue

            if self.global_ndarray is not None:
                self.global_ndarray = np.concatenate((self.global_ndarray, indata), dtype=np.int16)
            else:
                self.global_ndarray = indata

            # concatenate buffers if the end of the current buffer is not silent and if the chunksize is under 5
            if (np.average((indata_flattened[-100:-1])) > self.SILENCE_THRESHOLD / 15 and len(indata_flattened) / 16000 < 2) or time.monotonic() - start_time < duration:
                continue
            else:
                self.temp_ndarray = self.global_ndarray[self.global_ndarray != 0]
                self.global_ndarray = None
                return self.temp_ndarray
        
    async def set_silence_threshold(self):
        """
        This asynchronous method dynamically adjusts the silence threshold based on the loudness of the initial 
        audio input. It processes a predefined duration of audio to calculate an average loudness value, which 
        is then used to set the silence threshold. This adjustment is crucial for optimizing subsequent voice 
        activity detection and ensuring the system's sensitivity is tailored to the current environment's noise level. 
        A warning is issued if the calculated threshold is exceptionally high, indicating potential issues with 
        microphone input levels or environmental noise.
        """
        blocks_processed = 0
        loudness_values = []

        async for indata, _ in self.generate():
            blocks_processed += 1
            indata_flattened = abs(indata.flatten())

            # Compute loudness over first few seconds to adjust silence threshold
            loudness_values.append(np.mean(indata_flattened))

            # Stop recording after ADJUSTMENT_TIME seconds
            if blocks_processed >= self.ADJUSTMENT_TIME * self.SAMPLERATE / self.BLOCKSIZE:
                self.SILENCE_THRESHOLD = int(np.mean(loudness_values) * self.SILENCE_RATIO)
                break
            
        print(f'\nSet SILENCE_THRESHOLD to {self.SILENCE_THRESHOLD}\n')
        if self.SILENCE_THRESHOLD > 3000:
            warnings.warn(f'SILENCE_THRESHOLD is {self.SILENCE_THRESHOLD}, which is very high. This may cause unprecise results.')
        elif self.SILENCE_THRESHOLD < 1000:
            warnings.warn(f'SILENCE_THRESHOLD is {self.SILENCE_THRESHOLD}, which is very low. This may cause unprecise results.')
            self.SILENCE_THRESHOLD = 1000