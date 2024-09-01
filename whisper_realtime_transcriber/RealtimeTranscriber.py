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
    inputstream_generator : InputStreamGenerator
        The generator to be used for streaming audio.
    asr_model : WhisperModel
        The whisper model to be used for inference.
    continuous : bool
        Whether to generate audio data conituously or not. (default is True)
    verbose : bool
        Whether to print the additional information to the console or not. (default is True)
    func : Callable
        A specified function that is doing something with the transcriptions. (default is the builtin print function)

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

    def __init__(self, inputstream_generator: InputStreamGenerator, asr_model: WhisperModel, continuous: bool = True, verbose: bool = True, func: t.Callable = print):
        self._inputstream_generator = inputstream_generator
        self._asr_model = asr_model
        
        self._inputstream_generator.verbose, self._asr_model.verbose = verbose, verbose
        self._inputstream_generator.continuous, self._asr_model.continuous = continuous, continuous
        
        self.func = func
        
    def create_tasks(self) -> t.Tuple[t.AsyncGenerator, t.AsyncGenerator]:
        inputstream_task = asyncio.create_task(self._inputstream_generator.process_audio())
        transcribe_task = asyncio.create_task(self._asr_model.run_inference())
        return inputstream_task, transcribe_task

    async def execute_event_loop(self) -> None:
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
            
            except Exception as e:
                print(f"An error occurred: {e}")
                break
        
            finally:
                inputstream_task.cancel()
                transcribe_task.cancel()
                await asyncio.gather(inputstream_task, transcribe_task, return_exceptions=True)
