import torch
import asyncio
import typing as t
import numpy as np
import os
import shutil

from dataclasses import dataclass, field, asdict

from whisper_realtime_transcriber.utils.utils import set_device
from whisper_realtime_transcriber.InputStreamGenerator import InputStreamGenerator
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import logging

logging.set_verbosity_error()

@dataclass
class ModelOutput:
    logits: list[torch.Tensor]
    transcriptions: list[str]


@dataclass
class ModelArguments:
    """
    Arguments for creating the whisper model.
    """
    model_id: t.Optional[str] = field(
        default=None,
        metadata={"help": "The model id to be used for loading the model."}
    )
    model_size: str = field(
        default="small",
        metadata={"help": "The size of the model to be used for inference."}
        # choices=["small", "medium", "large-v3"]
    )
    device: str = field(
        default="cpu",
        metadata={"help": "The device to be used for inference."}
        # choices=["cpu", "cuda", "mps"]
    )
    continuous: bool = field(
        default=True,
        metadata={"help": "Whether to generate audio data continuously or not."}
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print the model outputs to the console or not."}
    )


class WhisperModel:
    """
    Loading and using the specified whisper model.

    Methods
    -------
    run_inference()
        Runs the inference of the model.
    """

    def __init__(
        self,
        inputstream_generator: InputStreamGenerator,
        model_args: ModelArguments
    ):
        self._inputstream_generator = inputstream_generator

        self._continuous = model_args.continuous

        self._device = set_device(model_args.device)

        self._working: bool = False

        self._torch_dtype = torch.float16 if self._device == torch.device("cuda") else torch.float32
        if self._device == torch.device("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True

        self._load_model(model_args.model_size, model_args.model_id)

        self.audio_data: np.ndarray = np.empty(0, dtype="int16")

        self._output = ModelOutput([torch.empty(0, 1)], [""])

        # Check if generator samplerate matches models samplerate
        if self._inputstream_generator.samplerate != self._processor.feature_extractor.sampling_rate:
            self._inputstream_generator.samplerate = self._processor.feature_extractor.sampling_rate

        self._verbose = model_args.verbose

    def _load_model(self, model_size: str, model_id: t.Optional[str]) -> None:
        if model_id is None:
            self.available_model_sizes = ["small", "medium", "large-v3"]

            self._model_size = "large-v3" if model_size == "large" else model_size

            if model_size not in self.available_model_sizes:
                print(f"Model size not supported. Defaulting to {self._model_size}.")

            self._model_id = (
                f"distil-whisper/distil-{self._model_size}.en"
                if self._model_size in self.available_model_sizes[:2]
                else f"distil-whisper/distil-{self._model_size}"
            )
        else:
            self._model_id = model_id

        self._speech_model = WhisperForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self._device)

        self._processor = WhisperProcessor.from_pretrained(self._model_id)

    async def run_inference(self) -> dict:
        """
        Runs the speech recognition inference in a loop, processing audio input as it becomes available.

        This method continuously monitors an event (`data_ready_event`) to determine when audio data is ready for
        transcription. Once the data is ready, it performs the transcription using the `_transcribe` method and
        computes performance metrics, including the real-time factor (RTF).

        Workflow:
            1. **Wait for Data**: The method waits until the `data_ready_event` is set, indicating that audio data is
               ready for processing.

            2. **Start Timer**: The method records the start time to measure the duration of the transcription process.

            3. **Transcription**: It calls the `_transcribe()` method to perform the actual speech recognition.

            4. **Check Continuous Mode**: If `self.continuous` is `False`, the method clears the `data_ready_event`
               and returns the transcription result. This means the loop exits after the first transcription.

            5. **Compute Metrics**:
                - **Audio Duration**: Calculates the duration of the audio input using the length of the audio data
                  and the sample rate.
                - **Transcription Duration**: Measures how long the transcription took.
                - **Real-Time Factor (RTF)**: The ratio of transcription time to audio duration. An RTF > 1 indicates
                  that the transcription took longer than the audio duration.

            6. **Clear Event**: Clears the `data_ready_event` after processing the data to prepare for the next round of input.

            7. **Verbose Output**: If `self.verbose` is `True`, it prints the transcription results using
               `_print_transcriptions()`.

            8. **Real-Time Factor Warning**: If the RTF > 1 and the system is not in a memory-safe mode, it warns the user
               that transcription took longer than the length of the audio input and suggests using a smaller model or
               adjusting configuration settings.

        Returns:
            str: The transcription result if `self.continuous` is `False`. In continuous mode, this method does not return
            until the loop is broken externally.
        """
        while True:
            await self._inputstream_generator.data_ready_event.wait()

            if not self._working:
                await self._transcribe()
            else:
                continue

            if self._inputstream_generator.complete_phrase_event.is_set() or not self._continuous:
                self.audio_data: np.ndarray = np.empty(0, dtype=np.int16)
                self._output.logits.append(torch.empty(0, 1))
                self._output.transcriptions.append("")
                self._inputstream_generator.data_ready_event.clear()
                self._inputstream_generator.complete_phrase_event.clear()

            if not self._continuous:
                self._output.transcriptions.remove("")
                self._output.logits.remove(torch.empty(0, 1))
                return asdict(self._output)

            self._inputstream_generator.data_ready_event.clear()

            if not self._verbose:
                continue

            asyncio.to_thread(self._print_transcriptions)

    async def _transcribe(self) -> None:
        """
        Main logic for running the actual inference on the models.
        """
        self._working = True

        a1 = self.audio_data
        a2 = self._inputstream_generator.temp_ndarray

        # Convert raw audio data to feasible input for the model.
        self.audio_data = (
            np.concatenate((a1, a2), axis=0, dtype="int16")
            if self.audio_data is not None
            else self._inputstream_generator.temp_ndarray
        )

        waveform = self.audio_data.flatten().astype(np.float32) / 32768.0
        waveform = torch.from_numpy(waveform)

        inputs = self._processor(
            waveform,
            sampling_rate=self._inputstream_generator.samplerate,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device, dtype=self._torch_dtype)

        # https://github.com/huggingface/transformers/pull/33145
        # Make prediction on the audio data.
        generated_ids = await asyncio.to_thread(
            self._speech_model.generate,
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            return_timestamps=False,
            pad_token_id=self._processor.tokenizer.pad_token_id,
            eos_token_id=self._processor.tokenizer.eos_token_id,
        )
        transcript = await asyncio.to_thread(
            self._processor.batch_decode,
            generated_ids,
            skip_special_tokens=True,
            decode_with_timestamps=False,
        )

        self._output.transcriptions[-1] = transcript[-1].strip()
        self._output.logits[-1] = generated_ids[-1]

        self._working = False

    def _print_transcriptions(self) -> None:
        """
        Prints the model transcription.
        """
        output = [transcription for transcription in self._output.transcriptions if transcription != ""]

        os.system("cls") if os.name == "nt" else os.system("clear")

        terminal_size = shutil.get_terminal_size()

        for transcription in output:
            words = transcription.split(" ")
            line_count = 0
            split_input = ""
            for word in words:
                line_count += len(word) + 1
                if line_count > terminal_size.columns:
                    split_input += "\n"
                    line_count = len(word) + 1
                    split_input += word
                    split_input += " "
                else:
                    split_input += word
                    split_input += " "

            print(split_input)
        print("", end="", flush=True)
