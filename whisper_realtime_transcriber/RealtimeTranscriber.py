import asyncio

from .WhisperModel import WhisperModel
from .InputStreamGenerator import InputStreamGenerator


class RealtimeTranscriber:
    def __init__(
        self,
        inputstream_generator: InputStreamGenerator,
        asr_model: WhisperModel,
        model_size: str = "small",
        device: str = "cpu",
        samplerate: int = 16000,
        blocksize: int = 4000,
        adjustment_time: int = 5,
        memory_safe: bool = True,
    ):
        self.model_size = model_size
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.adjustment_time = adjustment_time
        self.memory_safe = memory_safe

        self.inputstream_generator = inputstream_generator
        self.asr_model = asr_model

    def create_tasks(self):
        inputstream_task = asyncio.create_task(self.inputstream_generator.process_audio())
        transcribe_task = asyncio.create_task(self.asr_model.run_inference())
        return inputstream_task, transcribe_task

    async def start_event_loop(self):
        inputstream_task, transcribe_task = self.create_tasks()
        # Execute the tasks and catch exception
        try:
            await asyncio.gather(inputstream_task, transcribe_task)
        except asyncio.CancelledError:
            print("\nTranscribe task cancelled.")
            inputstream_task.cancel()
            transcribe_task.cancel()
