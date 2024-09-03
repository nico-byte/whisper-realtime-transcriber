# Whisper Realtime Transcriber

## Overview

This [repository](https://github.com/nico-byte/whisper-realtime-transcriber) contains the source code of a realtime transcriber for various [whisper](https://github.com/openai/whisper) models, published on [huggingface](https://github.com/huggingface/transformers).

## Prerequisites

Before you begin, make sure you meet the following prerequisites:

- [Python 3.10.12](https://www.python.org) installed on your machine.
- Microphone connected to your machine.

## Installation

1. **Install torch with CUDA support (optional)**
  - Follow the instructions [here](https://pytorch.org/get-started/locally/) and install version >=2.4.0

2. **Install the package:**
      ```bash
      pip install --upgrade whisper-realtime-transcriber
      ```

## Usage

After completing the installation, you can now use the transcriber:

  - Necessary imports
  ```python
  import asyncio

  from whisper_realtime_transcriber.InputStreamGenerator import InputStreamGenerator
  from whisper_realtime_transcriber.WhisperModel import WhisperModel
  from whisper_realtime_transcriber.RealtimeTranscriber import RealtimeTranscriber
  ```

  - Standard way - model and generator are initialized by the RealtimeTranscriber and all outputs get printed directly to the console.
  ```python
  transcriber = RealtimeTranscriber()

  asyncio.run(transcriber.execute_event_loop())
  ```

  - Executing a custom function inside the RealtimeTranscriber.
  ```python
  def print_transcription(some_transcription):
    print(some_transcription)
  
  # Specifying a function will set continuous to False - this will allow one
  # to execute a custom function during the coroutine, that is doing something with the transcriptions.
  # After the function finished it's work the coroutine will restart.
  transcriber = RealtimeTranscriber(func=print_transcription)
    
  asyncio.run(transcriber.execute_event_loop())
  ```

  - Loading the InputStreamGenerator and/or Whisper Model with custom values.
  ```python
  inputstream_generator = InputStreamGenerator(samplerate=8000, blocksize=2000, min_chunks=2)
  asr_model = WhisperModel(inputstream_generator, model_id="openai/whisper-tiny", device="cuda")

  transcriber = RealtimeTranscriber(inputstream_generator, asr_model)

  asyncio.run(transcriber.execute_event_loop())
  ```

Feel free to reach out if you encounter any issues or have questions!

## How it works

- The transcriber consists of two modules: a Inputstream Generator and a Whisper Model.
- The implementation of the Inputstream Generator is based on this [implemantation](https://github.com/tobiashuttinger/openai-whisper-realtime).
- The Inputstream Generator reads the microphone input and passes it to the Whisper Model. The Whisper Model then generates the transcription.
- This is happening in an async event loop so that the Whsiper Model can continuously generate transcriptions from the provided audio input, generated and processed by the Inputstream Generator.
- On a machine with a 12GB Nvidia RTX 3060 the [distilled large-v3](https://github.com/huggingface/distil-whisper) model runs at a realtime-factor of about 0.4, this means 10s of audio input get transcribed in 4s - the longer the input the bigger is the realtime-factor.

## ToDos

- Add functionality to transcribe from audio files.
- Get somehow rid of the hallucinations of the whisper models when no voice is active.