import numpy as np
import asyncio
import sounddevice as sd

from typing import Tuple, List
from async_class import AsyncClass


class LiveAudioTranscriber(AsyncClass):
    """
    Provides a class for providing the live audio transcription.
    ----------------------------------------------------------------------------------
    Paramters
    ----------
    samplerate: int
        The samplerate to use for the input stream. Default: 16000
    blocksize: int
        The size of the blocks to use for the input stream. Default: 8000
    silence_ratio: int
        The max amount of silent values in one block. Default: 3500
    ajustment_time: int
        The duration used for generating the silence_threshold. Default: 5
    silence_threshold: int
        If it is not desired to auto generate a silence_threshold, set this value to a desired value. Default: set_silence_threshold()
    """
    async def __ainit__(self, samplerate: int=None, blocksize: int=None, silence_ratio: int=None, adjustment_time: int=None, silence_threshold: int=None):
        self.SAMPLERATE = 16000 if samplerate is None else samplerate
        self.BLOCKSIZE = 8000 if blocksize is None else blocksize
        self.SILENCE_RATIO = 3500 if silence_ratio is None else silence_ratio
        self.ADJUSTMENT_TIME = 5 if adjustment_time is None else adjustment_time
        
        self.SILENCE_THRESHOLD = silence_threshold
        
        if self.SILENCE_THRESHOLD is None:
            await self.set_silence_threshold()
        
        self.global_ndarray: np.ndarray = None
        self.temp_ndarray: np.ndarray = None
        
        self.silence_counter: int = 0
        
        print(f"Checked inputstream parameters: \n\
            samplerate: {self.SAMPLERATE}\n\
                blocksize: {self.BLOCKSIZE}\n\
                    SILENCE_RATIO: {self.SILENCE_RATIO}")

    async def generate(self):
        """Generates an input stream of audio data asynchronously.
    
        This method sets up an asynchronous input stream using the `sounddevice` library, with a specified sample rate, block size, and callback function. 
        The callback function puts the incoming audio data and status into an asynchronous queue, which is then yielded one block at a time.
    
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
    
    async def transcribe(self, model, loop_forever: bool, execution_interval: int) -> Tuple[List[str], List[str], List[str]]:
        """Transcribes live audio input using the provided Inference and InputStreamGenerator classes.
        
        This method listens for audio input, processes it in blocks, and runs inference on the accumulated audio data. 
        The transcription results are either printed continuously or returned as a tuple.
        
        Args:
            model (Any): The Model class to use for inference.
            loop_forever (bool): If True, the method will continuously print transcription results.
            execution_interval (int): The number of seconds between each inference run.
        
        Returns:
            Tuple[List[str], List[str], List[str]]: If loop_forever == True - A tuple containing the transcription, original tokens, and processed tokens.
        """
        print("Listening...")
        
        async for indata, _ in self.generate():
            indata_flattened: np.ndarray = abs(indata.flatten())
            
            if self.global_ndarray is not None and self.silence_counter >= (self.SAMPLERATE / self.BLOCKSIZE) * execution_interval:
                self.temp_ndarray = self.global_ndarray.copy()
                self.temp_ndarray = self.temp_ndarray.flatten().astype(np.float32) / 32768.0
                transcript, original_tokens, processed_tokens =  await model.run_inference(self.temp_ndarray)
                
                if loop_forever:
                    await self.print_transcriptions()
                    self.global_ndarray: np.ndarray = None
                else:
                    return transcript, original_tokens, processed_tokens
                
            # discard buffers that contain mostly silence
            if len(np.nonzero(indata_flattened > self.SILENCE_THRESHOLD)[0]) < self.SILENCE_RATIO:
                self.silence_counter += 1
                continue
            if self.global_ndarray is not None:
                self.silence_counter = 0
                self.global_ndarray = np.concatenate((self.global_ndarray, indata), dtype='int16')
            else:
                self.global_ndarray = indata
            # concatenate buffers if the end of the current buffer is not silent and if the chunksize is under 5
            if (np.average((indata_flattened[-100:-1])) > self.SILENCE_THRESHOLD):
                continue
        
    async def print_transcriptions(self):
        """Prints the current transcript, ensuring that the output is formatted with a maximum line length of 77 characters. 
        
        If the addition of the current transcript would exceed the line length, a new line is started.
        """
        char_limit: int = 77  # The character limit after which a new line should start
        current_line_length: int = 0  # Current length of the line being printed

        # Calculate the new line length if the update is added
        new_line_length: int = current_line_length + len(self.transcript)
        if new_line_length > char_limit:
            print()  # Start a new line if the limit is exceeded
            current_line_length = 0  # Reset the current line length
        print(self.transcript + " ", end='', flush=True)  # Print the update without a newline, flush to ensure it's displayed
        current_line_length += len(self.transcript) # Update the current line length
        
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
