import asyncio
import typing as t

from .WhisperModel import WhisperModel
from .InputStreamGenerator import InputStreamGenerator


class RealtimeTranscriber:
    """
    Loading and using the RealtimeTranscriber.

    Parameters
    ----------
    inputstream_generator : InputStreamGenerator
        The generator to be used for streaming audio.
    asr_model : WhisperModel
        The whisper model to be used for inference.
    verbose : bool
        Whether to print the additional information to the console or not. (default is True)

    Attributes
    ----------
    verbose : bool
        Where the boolean to decide to print the model outputs is stored.

    Methods
    -------
    create_tasks()
        Creating the tasks responsible for generating the audio data and inference.
    start_event_loop()
        Starting the event loop responsible for realtime transcribing.
    """

    def __init__(self, inputstream_generator: InputStreamGenerator, asr_model: WhisperModel, verbose: bool = True):
        self._inputstream_generator = inputstream_generator
        self._asr_model = asr_model

        self._inputstream_generator.verbose, self._asr_model.verbose = verbose, verbose

    def create_tasks(self) -> t.Tuple[t.AsyncGenerator, t.AsyncGenerator]:
        inputstream_task = asyncio.create_task(self._inputstream_generator.process_audio())
        transcribe_task = asyncio.create_task(self._asr_model.run_inference())
        return inputstream_task, transcribe_task

    async def start_event_loop(self) -> None:
        inputstream_task, transcribe_task = self.create_tasks()
        # Execute the tasks and catch exception
        try:
            await asyncio.gather(inputstream_task, transcribe_task)
        except asyncio.CancelledError:
            print("\nTranscribe task cancelled.")
            inputstream_task.cancel()
            transcribe_task.cancel()
