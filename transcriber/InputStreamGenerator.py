import numpy as np
import asyncio
import sys

try:
    import sounddevice as sd
except OSError as e:
    print(e)
    print(
        "If `GLIBCXX_x.x.x' not found, try installing it with: conda install -c conda-forge libstdcxx-ng=12"
    )
    sys.exit()

from utils.decorators import sync_timer


class InputStreamGenerator:
    @sync_timer(print_statement="Loaded inputstream generator", return_some=False)
    def __init__(
        self,
        samplerate=16000,
        blocksize=4000,
        adjustment_time=5,
        min_chunks=8,
        memory_safe=True,
    ):
        """
        :param samplerate (int): the samplerate to use for the audio input
        :param blocksize (int): the blocksize to use for the audio input
        :param adjustment_time (int): the time to wait for adjusting the silence_threshold
        """
        self.SAMPLERATE = samplerate
        self.BLOCKSIZE = blocksize
        self.ADJUSTMENT_TIME = adjustment_time
        self.min_chunks = min_chunks
        self.memory_safe = memory_safe

        self.global_ndarray: np.ndarray = None
        self.temp_ndarray: np.ndarray = None

        self.data_ready_event = asyncio.Event()

    async def _generate(self):
        """
        Generate audio chunks of size of the blocksize and yield them.
        """
        q_in = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(in_data, _, __, state):
            loop.call_soon_threadsafe(q_in.put_nowait, (in_data.copy(), state))

        stream = sd.InputStream(
            samplerate=self.SAMPLERATE,
            channels=1,
            dtype="int16",
            blocksize=self.BLOCKSIZE,
            callback=callback,
        )
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata, status

    async def process_audio(self):
        """
        Process the audio chunks and store them for the transcriber.
        """
        await self._set_silence_threshold()

        print("Listening...")

        async for indata, _ in self._generate():
            indata_flattened: np.ndarray = abs(indata.flatten())

            # discard buffers that contain mostly silence
            if (
                (np.percentile(indata_flattened, 10) <= self.SILENCE_THRESHOLD)
                and self.global_ndarray is None
            ) or (self.memory_safe and self.data_ready_event.is_set()):
                continue
            if self.global_ndarray is not None:
                self.global_ndarray = np.concatenate(
                    (self.global_ndarray, indata), dtype="int16"
                )
            else:
                self.global_ndarray = indata
            # concatenate buffers if the end of the current buffer is not silent
            if (
                np.percentile(indata_flattened[-100:-1], 10) > self.SILENCE_THRESHOLD
            ) or self.data_ready_event.is_set():
                continue
            elif len(self.global_ndarray) / self.BLOCKSIZE >= self.min_chunks:
                self.temp_ndarray = self.global_ndarray.copy()
                self.temp_ndarray = (
                    self.temp_ndarray.flatten().astype(np.float32) / 32768.0
                )

                self.global_ndarray = None
                self.data_ready_event.set()
            else:
                continue

    async def _set_silence_threshold(self):
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
            if blocks_processed >= self.ADJUSTMENT_TIME * (
                self.SAMPLERATE / self.BLOCKSIZE
            ):
                self.SILENCE_THRESHOLD = float(np.percentile(loudness_values, 20))
                break

        print(f"\nSet SILENCE_THRESHOLD to {self.SILENCE_THRESHOLD}\n")
