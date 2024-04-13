import numpy as np
import sounddevice as sd
import asyncio

from async_class import AsyncClass


class InputStreamGenerator(AsyncClass):
    """
    https://github.com/tobiashuttinger/openai-whisper-realtime/blob/main/openai-whisper-realtime.py \n
    Provides a class for generating an input stream of audio data.
    --------------------------------------------------------------
    Parameters
    ----------
    samplerate: int
        The samplerate to use for the input stream.
    blocksize: int
        The size of the blocks to use for the input stream.
    silence_ratio: int
        The max amount of silent values in one block.
    ajustment_time: int
        The duration used for generating the silence_threshold.
    silence_threshold: int
        If it is not desired to auto generate a silence_threshold, set this value to a desired value.
    """
    async def __ainit__(self, samplerate: int, blocksize: int, silence_ratio: int, adjustment_time: int, silence_threshold: int):  
        self.SAMPLERATE = samplerate
        self.BLOCKSIZE = blocksize
        self.SILENCE_RATIO = silence_ratio
        self.ADJUSTMENT_TIME = adjustment_time
        self.SILENCE_THRESHOLD = silence_threshold
        
        self.global_ndarray: np.ndarray = None
        self.temp_ndarray: np.ndarray = None

    async def generate(self):
        """Generates an input stream of audio data asynchronously.
    
        This method sets up an asynchronous input stream using the `sounddevice` library, with a specified sample rate, block size, and callback function. The callback function puts the incoming audio data and status into an asynchronous queue, which is then yielded one block at a time.
    
        Yields:
            Tuple[numpy.ndarray, int]: A tuple containing the audio data block and the status of the input stream.
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
    
    async def set_silence_threshold(self):
        """Automatically sets the silence threshold for the input stream based on the first few seconds of audio data.
    
        This method processes incoming audio data from the input stream, computes the average loudness over the first few seconds, and sets the `SILENCE_THRESHOLD` attribute based on the computed average loudness and the `SILENCE_RATIO` parameter.
    
        The method continues processing audio data until the `ADJUSTMENT_TIME` duration has elapsed, at which point it sets the `SILENCE_THRESHOLD` and exits.
        """
        blocks_processed: int = 0
        loudness_values: list = []

        async for indata, _ in self.generate():
            blocks_processed += 1
            indata_flattened: np.ndarray = abs(indata.flatten())

            # Compute loudness over first few seconds to adjust silence threshold
            loudness_values.append(np.mean(indata_flattened))

            # Stop recording after ADJUSTMENT_TIME seconds
            if blocks_processed >= self.ADJUSTMENT_TIME * self.SAMPLERATE / self.BLOCKSIZE:
                self.SILENCE_THRESHOLD = int((np.mean(loudness_values) * self.SILENCE_RATIO) / 15)
                break
            
        print(f'\nSet SILENCE_THRESHOLD to {self.SILENCE_THRESHOLD}\n')