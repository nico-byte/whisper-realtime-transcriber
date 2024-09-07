import asyncio
import sys
import typing as t

from .WhisperModel import WhisperModel
from .InputStreamGenerator import InputStreamGenerator


class RealtimeTranscriber:
    """
    Loading and using the RealtimeTranscriber.

    Parameters
    ----------
    inputstream_generator : Optional[InputStreamGenerator]
        The generator to be used for streaming audio.
    asr_model : Optional[WhisperModel]
        The whisper model to be used for inference.
    continuous : bool
        Whether to generate audio data conituously or not. (default is True)
    memory_safe: bool
        Whether to pause the generation audio data during the inference of the asr model or not. (default is True)
    device : str
        The device to be used for inference. ("cpu", "cuda", "mps" - default is "cpu")
    verbose : bool
        Whether to print the additional information to the console or not. (default is True)
    func : Callable
        A specified function that is doing something with the transcriptions. (default is the builtin print function)

    Attributes
    ----------
    continuous : bool
        Where the boolean to decide if the event loop runs continuously.
    verbose : bool
        Where the boolean to decide to print the model outputs is stored.

    Methods
    -------
    create_tasks()
        Creating the tasks responsible for generating the audio data and inference.
    start_event_loop()
        Starting the event loop responsible for realtime transcribing.
    """

    def __init__(
        self,
        inputstream_generator: t.Optional[InputStreamGenerator] = None,
        asr_model: t.Optional[WhisperModel] = None,
        continuous: bool = True,
        memory_safe: bool = True,
        device: str = "cpu",
        verbose: bool = True,
        func: t.Callable = None,
    ):
        self._inputstream_generator = inputstream_generator if inputstream_generator is not None else self._init_generator()
        self._asr_model = asr_model if asr_model is not None else self._init_asr_model()

        self._configure(verbose, memory_safe, device, continuous)
        self.func = func or print

    def _default_inputstream_generator(self) -> InputStreamGenerator:
        # Create and return the default InputStreamGenerator
        return InputStreamGenerator()

    def _default_asr_model(self) -> WhisperModel:
        # Create and return the default WhisperModel
        return WhisperModel(self._inputstream_generator)
    
    def _configure(self, verbose: bool, memory_safe: bool, device: str, continuous: bool):
        self._inputstream_generator.verbose = verbose
        self._asr_model.verbose = verbose
        self._inputstream_generator.memory_safe = memory_safe
        self._asr_model.device = device

        self._inputstream_generator.continuous = continuous
        self._asr_model.continuous = continuous

    def create_tasks(self) -> t.Tuple[t.AsyncGenerator, t.AsyncGenerator]:
        """
        Creates and returns two asynchronous tasks to handle audio processing and speech recognition.

        This method sets up two asynchronous tasks using `asyncio.create_task()`:

        1. `inputstream_task`: This task processes the audio input stream using the `_inputstream_generator` object.
        2. `transcribe_task`: This task runs the automatic speech recognition (ASR) inference using the `_asr_model` object.

        These tasks are returned as a tuple, allowing them to be awaited or managed concurrently.

        Returns:
            tuple: A tuple containing the following two asynchronous tasks:
                - `inputstream_task` (asyncio.Task): The task responsible for processing the audio stream.
                - `transcribe_task` (asyncio.Task): The task responsible for running ASR inference.
        """
        inputstream_task = asyncio.create_task(self._inputstream_generator.process_audio())
        transcribe_task = asyncio.create_task(self._asr_model.run_inference())
        return inputstream_task, transcribe_task

    async def execute_event_loop(self) -> None:
        """
        Continuously executes an event loop to process audio input and perform transcription.

        This method runs an infinite loop that continuously creates and executes tasks for processing audio and
        transcribing speech. It handles different types of exceptions to ensure proper task management and graceful shutdown.

        Workflow:
            1. It creates two asynchronous tasks using the `create_tasks()` method:
                - `inputstream_task`: Processes the audio input stream.
                - `transcribe_task`: Performs the transcription using an ASR model.

            2. Both tasks are executed concurrently using `asyncio.gather()`.
               The transcription result is passed to `self.func()` for further processing.

            3. Exception Handling:
                - **CancelledError**: If the task is cancelled, both tasks are cancelled and the loop breaks.
                - **KeyboardInterrupt**: If interrupted by the user (e.g., Ctrl+C), the program exits.

            4. Finally block: Ensures that both tasks are cancelled, and any pending tasks are awaited even if an exception occurs.

        Returns:
            None
        """
        while True:
            inputstream_task, transcribe_task = self.create_tasks()

            # Execute the tasks and catch exceptions
            try:
                _, transcription = await asyncio.gather(inputstream_task, transcribe_task)
                self.func(transcription)

            except asyncio.CancelledError:
                print("\nTranscribe task cancelled.")
                inputstream_task.cancel()
                transcribe_task.cancel()

                await asyncio.gather(inputstream_task, transcribe_task, return_exceptions=True)
                break

            except KeyboardInterrupt:
                sys.exit("\nInterrupted by user")

            finally:
                inputstream_task.cancel()
                transcribe_task.cancel()
                await asyncio.gather(inputstream_task, transcribe_task, return_exceptions=True)
                await asyncio.sleep(0.1)
