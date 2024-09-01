import numpy as np
import asyncio
import sys
import typing as t

try:
    import sounddevice as sd
except OSError as e:
    print(e)
    print("If `GLIBCXX_x.x.x' not found, try installing it with: conda install -c conda-forge libstdcxx-ng=12")
    sys.exit()
    

class InputStreamGenerator:
    """
    Loading and using the InputStreamGenerator.

    Parameters
    ----------
    samplerate : int
        The specified samplerate of the audio data. (default is 16000)
    blocksize : int
        The size of each individual audio chunk. (default is 4000)
    adjustment_time : int
        The adjustment_time for setting the silence threshold. (default is 5)
    min_chunks : int
        The minimum number of chunks to be generated, before feeding it into the asr model. (default is 6)
    continuous : bool
        Whether to generate audio data conituously or not. (default is True)
    memory_safe: bool
        Whether to pause the generation audio data during the inference of the asr model or not. (default is True)
    verbose : bool
        Whether to print the additional information to the console or not. (default is True)

    Attributes
    ----------
    samplerate : int
        The samplerate of the generated audio data.
    temp_ndarray : np.ndarray
        Where the generated audio data is stored.
    data_ready_event : asyncio.Event
        Boolean to tell the InputStreamGenerator, that the asr model is busy or not.
    memory_safe: bool
        If True, InputStreamGenerator will pause the collection of audio data during model inference.
    verbose : bool
        Where the boolean to decide to print the model outputs is stored.

    Methods
    -------
    process_audio()
        Processes the incoming audio data.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        blocksize: int = 4000,
        adjustment_time: int = 5,
        min_chunks: int = 6,
        continuous: bool = True,
        memory_safe: bool = True,
        verbose: bool = True,
    ):
        self.samplerate = samplerate
        self._blocksize = blocksize
        self._adjustment_time = adjustment_time
        self._min_chunks = min_chunks
        self.continuous = continuous
        self.memory_safe = memory_safe
        self.verbose = verbose

        self._global_ndarray: np.ndarray = None
        self.temp_ndarray: np.ndarray = None
        
        self._silence_threshold = None

        self.data_ready_event = asyncio.Event()
        
    async def _generate(self) -> t.AsyncGenerator:
        """
        Generate audio chunks of size of the blocksize and yield them.
        """
        q_in = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(in_data, _, __, state):
            loop.call_soon_threadsafe(q_in.put_nowait, (in_data.copy(), state))

        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
            blocksize=self._blocksize,
            callback=callback,
        )
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata, status

    async def process_audio(self) -> None:
        """
        Process the audio chunks and store them for the transcriber.
        """
        if self._silence_threshold is None:
            await self._set_silence_threshold()
        
        print("Listening...")

        async for indata, _ in self._generate():
            indata_flattened: np.ndarray = abs(indata.flatten())

            # discard buffers that contain mostly silence
            if ((np.percentile(indata_flattened, 10) <= self._silence_threshold) and self._global_ndarray is None) or (
                self.memory_safe and self.data_ready_event.is_set()
            ):
                continue
            if self._global_ndarray is not None:
                self._global_ndarray = np.concatenate((self._global_ndarray, indata), dtype="int16")
            else:
                self._global_ndarray = indata
            # concatenate buffers if the end of the current buffer is not silent
            if (np.percentile(indata_flattened[-100:-1], 10) > self._silence_threshold) or self.data_ready_event.is_set():
                continue
            elif len(self._global_ndarray) / self._blocksize >= self._min_chunks:
                self.temp_ndarray = self._global_ndarray.copy()
                self.temp_ndarray = self.temp_ndarray.flatten().astype(np.float32) / 32768.0

                self._global_ndarray = None
                self.data_ready_event.set()
                if not self.continuous:
                    return None
            else:
                continue
        
    async def _set_silence_threshold(self) -> None:
        """
        Automatically adjust the silence threshold based on the 20th percentile of the loudness of the input.
        """
        blocks_processed: int = 0
        loudness_values: list = []

        async for indata, _ in self._generate():
            blocks_processed += 1
            indata_flattened: np.ndarray = abs(indata.flatten())

            # Compute loudness over first few seconds to adjust silence threshold
            loudness_values.append(np.mean(indata_flattened))

            # Stop recording after ADJUSTMENT_TIME seconds
            if blocks_processed >= self._adjustment_time * (self.samplerate / self._blocksize):
                self._silence_threshold = float(np.percentile(loudness_values, 20))
                break

        if self.verbose:
            print(f"Set SILENCE_THRESHOLD to {self._silence_threshold}\n")
