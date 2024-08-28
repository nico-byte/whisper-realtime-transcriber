import torch
import asyncio
import time
import string

from utils.utils import preprocess_text, set_device
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from utils.decorators import sync_timer


class DistilWhisper:
    @sync_timer(print_statement="Loaded distilled whisper model", return_some=False)
    def __init__(
        self,
        inputstream_generator,
        model_size="small",
        punctuate_truecase=False,
        device="cpu",
    ):
        """
        :param inputstream_generator: the generator to use for streaming audio
        :param model_size (str): the size of the model to use for inference
        :param language (str): the language to use for tokenizing the model output
        :param device (str): the device to use for inference
        """
        self.available_model_sizes = ["small", "medium", "large-v3"]

        self.model_size = model_size
        self.model_size = "large-v3" if model_size == "large" else self.model_size

        self.model_id = (
            f"distil-whisper/distil-{self.model_size}.en"
            if self.model_size in self.available_model_sizes[:2]
            else f"distil-whisper/distil-{self.model_size}"
        )

        self.inputstream_generator = inputstream_generator

        if model_size not in self.available_model_sizes:
            print(f"Model size not supported. Defaulting to {self.model_size}.")

        self.device = set_device(device)

        self.torch_dtype = (
            torch.float16 if self.device == torch.device("cuda") else torch.float32
        )
        torch.backends.cuda.matmul.allow_tf32 if self.device == torch.device(
            "cuda"
        ) else None

        self.transcript: str = ""
        self.full_sentences: str = ""
        self.partial_sentence: str = ""

        self.punctuate_truecase = punctuate_truecase
        self.remove_punct_map = {
            ord(char): None
            for char in string.punctuation
            if char not in ["ä", "ö", "ü", "ß"]
        }

        self.speech_model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.processor = WhisperProcessor.from_pretrained(self.model_id)

        # Check if generator samplerate matches models samplerate
        if (
            self.inputstream_generator.SAMPLERATE
            != self.processor.feature_extractor.sampling_rate
        ):
            self.inputstream_generator.SAMPLERATE = (
                self.processor.feature_extractor.sampling_rate
            )
            
    async def run_inference(self):
        """
        Main logic for calling an inference run, computing the real-time factor and printing the transcription.
        """
        while True:
            await self.inputstream_generator.data_ready_event.wait()
            start_time = time.perf_counter()

            await self._transcribe()

            # Compute the duration of the audio input and comparing it to the duration of inference.
            audio_duration = (
                len(self.inputstream_generator.temp_ndarray)
                / self.inputstream_generator.SAMPLERATE
            )

            if self.punctuate_truecase:
                await self._print_transcriptions() if self.full_sentences else None
            else:
                await self._print_transcriptions()
                self.transcript = ""

            self.inputstream_generator.data_ready_event.clear()

            transcription_duration = time.perf_counter() - start_time
            realtime_factor = transcription_duration / audio_duration

            # Warn the user when real-time factor>1
            if realtime_factor > 1 and not self.inputstream_generator.memory_safe:
                print(
                    f"\nTranscription took longer ({transcription_duration:.3f}s) than length of input in seconds ({audio_duration:.3f}s)."
                )
                print(
                    f"Real-Time Factor: {realtime_factor:.3f}, try to use a smaller model."
                )

    async def _transcribe(self):
        """
        Main logic for running the actual inference on the models.
        """
        # Convert raw audio data to feasible input for the model.
        waveform = torch.from_numpy(self.inputstream_generator.temp_ndarray)

        inputs = self.processor(
            waveform,
            sampling_rate=self.inputstream_generator.SAMPLERATE,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device, dtype=self.torch_dtype)

        # https://github.com/huggingface/transformers/pull/33145
        # Make prediction on the audio data.
        generated_ids = await asyncio.to_thread(
            self.speech_model.generate,
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            return_timestamps=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        transcript = await asyncio.to_thread(
            self.processor.batch_decode,
            generated_ids,
            skip_special_tokens=True,
            decode_with_timestamps=False,
        )

        self.transcript += transcript[0]
        await asyncio.to_thread(self._strip_transcript)
        if self.punctuate_truecase:
            self.full_sentences, self.partial_sentence = await asyncio.to_thread(
                preprocess_text, self.transcript
            )
            self.transcript = self.partial_sentence

    def _strip_transcript(self):
        self.transcript = self.transcript.lower()
        self.transript = self.transcript.translate(self.remove_punct_map).strip()
        self.transcript = self.transcript.replace(".", "")

    async def _print_transcriptions(self):
        """
        Prints the model trasncription.
        """
        output = self.full_sentences if self.punctuate_truecase else self.transcript

        char_limit: int = 77  # The character limit after which a new line should start
        current_line_length: int = 0  # Current length of the line being printed

        # Calculate the new line length if the update is added
        new_line_length: int = current_line_length + len(output)
        if new_line_length > char_limit:
            print()  # Start a new line if the limit is exceeded
            current_line_length = 0  # Reset the current line length
        print(
            output + " ", end="", flush=True
        )  # Print the update without a newline, flush to ensure it's displayed
        current_line_length += len(output)  # Update the current line length

    def get_models(self):
        print(f"Available models: {self.available_model_sizes}")
