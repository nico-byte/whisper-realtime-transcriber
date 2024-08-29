import torch
import asyncio
import time
import string

from utils.utils import preprocess_text, set_device
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import logging

from utils.decorators import sync_timer

logging.set_verbosity_error()


class WhisperModel:
    @sync_timer(print_statement="Loaded distilled whisper model", return_some=False)
    def __init__(self, inputstream_generator, model_id=None, model_size="small", punctuate_truecase=False, device="cpu", verbose=True):
        """
        :param inputstream_generator: the generator to use for streaming audio
        :param model_size (str): the size of the model to use for inference
        :param language (str): the language to use for tokenizing the model output
        :param device (str): the device to use for inference
        """
        self._inputstream_generator = inputstream_generator

        if model_id is None:
            self.available_model_sizes = ["small", "medium", "large-v3"]

            self._model_size = model_size
            self._model_size = "large-v3" if model_size == "large" else self._model_size

            if model_size not in self.available_model_sizes:
                print(f"Model size not supported. Defaulting to {self.model_size}.")

            self._model_id = (
                f"distil-whisper/distil-{self._model_size}.en"
                if self._model_size in self.available_model_sizes[:2]
                else f"distil-whisper/distil-{self._model_size}"
            )
        else:
            self._model_id = model_id

        self._device = set_device(device)

        self._torch_dtype = torch.float16 if self._device == torch.device("cuda") else torch.float32
        if self._device == torch.device("cuda"):
            torch.backends.cuda.matmul.allow_tf32

        self._temp_transcript: str = ""
        self.transcription: str = ""
        self._partial_transcript: str = ""

        self._punctuate_truecase = punctuate_truecase
        self._remove_punct_map = {ord(char): None for char in string.punctuation if char not in ["ä", "ö", "ü", "ß"]}

        self._speech_model = WhisperForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self._device)

        self._processor = WhisperProcessor.from_pretrained(self._model_id)

        # Check if generator samplerate matches models samplerate
        if self._inputstream_generator.SAMPLERATE != self._processor.feature_extractor.sampling_rate:
            self._inputstream_generator.SAMPLERATE = self.processor.feature_extractor.sampling_rate

        self.verbose = verbose

    async def run_inference(self):
        """
        Main logic for calling an inference run, computing the real-time factor and printing the transcription.
        """
        while True:
            await self._inputstream_generator.data_ready_event.wait()
            start_time = time.perf_counter()

            await self._transcribe()

            # Compute the duration of the audio input and comparing it to the duration of inference.
            audio_duration = len(self._inputstream_generator.temp_ndarray) / self._inputstream_generator.SAMPLERATE

            self._inputstream_generator.data_ready_event.clear()

            transcription_duration = time.perf_counter() - start_time
            realtime_factor = transcription_duration / audio_duration

            if not self.verbose:
                continue

            if self._punctuate_truecase:
                await self._print_transcriptions()
            else:
                await self._print_transcriptions()
                self._temp_transcript = ""

            # Warn the user when real-time factor>1
            if realtime_factor > 1 and not self._inputstream_generator.memory_safe:
                print(f"\nTranscription took longer ({transcription_duration:.3f}s) than length of input in seconds ({audio_duration:.3f}s).")
                print(
                    f"Real-Time Factor: {realtime_factor:.3f}, try to use a smaller model or increase the min_chunks option in the config file."
                )

    async def _transcribe(self):
        """
        Main logic for running the actual inference on the models.
        """
        # Convert raw audio data to feasible input for the model.
        waveform = torch.from_numpy(self._inputstream_generator.temp_ndarray)

        inputs = self._processor(
            waveform,
            sampling_rate=self._inputstream_generator.SAMPLERATE,
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

        self._temp_transcript += transcript[0]
        await asyncio.to_thread(self._strip_transcript)
        if self._punctuate_truecase:
            self.transcription, self._partial_transcript = await asyncio.to_thread(preprocess_text, self._temp_transcript)
            self._temp_transcript = self._partial_transcript

    def _strip_transcript(self):
        self._temp_transcript = self._temp_transcript.lower()
        self._temp_transript = self._temp_transcript.translate(self._remove_punct_map).strip()
        self._temp_transcript = self._temp_transcript.replace(".", "")

    async def _print_transcriptions(self):
        """
        Prints the model trasncription.
        """
        output = self.transcription if self._punctuate_truecase else self._temp_transcript

        char_limit: int = 77  # The character limit after which a new line should start
        current_line_length: int = 0  # Current length of the line being printed

        # Calculate the new line length if the update is added
        new_line_length: int = current_line_length + len(output)
        if new_line_length > char_limit:
            print()  # Start a new line if the limit is exceeded
            current_line_length = 0  # Reset the current line length
        print(output, end="", flush=True)  # Print the update without a newline, flush to ensure it's displayed
        current_line_length += len(output)  # Update the current line length
