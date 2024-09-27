import asyncio
import sys
import typing as t
from dataclasses import dataclass, asdict

from .WhisperModel import WhisperModel, ModelArguments
from .InputStreamGenerator import InputStreamGenerator, GeneratorArguments


class RealtimeTranscriber:
    """
    Loading and using the RealtimeTranscriber.

    Methods
    -------
    create_tasks()
        Creating the tasks responsible for generating the audio data and inference.
    start_event_loop()
        Starting the event loop responsible for realtime transcribing.
    """

    def __init__(
        self,
        model_args: t.Optional[ModelArguments] = None,
        generator_args: t.Optional[GeneratorArguments] = None,
        func: t.Callable[..., t.Any] = None,
    ):
        print(asdict(model_args))
        print(asdict(generator_args))
        self._generator = self._create_generator(generator_args) if generator_args is not None else self._create_generator(GeneratorArguments())
        self._asr_model = self._create_asr_model(model_args) if model_args is not None else self._create_asr_model(ModelArguments())

        self.func = func or print

    @staticmethod
    def _create_generator(generator_args: t.Optional[GeneratorArguments] = None) -> InputStreamGenerator:
        # Create and return the default InputStreamGenerator
        return InputStreamGenerator(generator_args)

    def _create_asr_model(self, model_args) -> WhisperModel:
        # Create and return the default WhisperModel
        return WhisperModel(self._generator, model_args)

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
        inputstream_task = asyncio.create_task(self._generator.process_audio())
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
                _, model_output = await asyncio.gather(inputstream_task, transcribe_task)
                await self.func(model_output.transcriptions)

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
