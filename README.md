# Whisper Realtime Transcriber

## Overview

This repository contains the source code of a realtime transcriber for various [whisper](https://github.com/openai/whisper) models, published on [huggingface](https://github.com/huggingface/transformers).

## Prerequisites

Before you begin, make sure you meet the following prerequisites:

- [Python 3.10.12](https://www.python.org) installed on your machine.
- Microphone connected to your machine.

## Installation Process

Follow the steps below to set up the project on your local machine:

1. **Clone the Project:**
  - Clone this repository to your local machine using `git`:
    ```bash
    git clone https://github.com/nico-byte/whisper-realtime-transcriber
    ```

2. **Enable CUDA (optional)**
  - If a CUDA device should be used for inference, PyTorch has to be installed with cuda support.
  - To achieve this, one must uncomment line 2 and 3 in the `requirements.in`file, and also comment the 4th line.

3. **Install dependencies:**
  - Either install all dependencies via `venv`:
    - Make the install script (if needed):
      ```bash
      chmod +x install.sh
      ```
    - Execute the install script:
      ```bash
      ./install.sh
      ```
  - Or install all dependencies via [Conda](https://anaconda.org):
    - Create a `conda` environment named `whisper-realtime`:
      ```bash
      conda create --name whisper-realtime python=3.10.12
      ```
    - Now activate the `conda` environment:
      ```bash
      conda activate whisper-realtime
      ```
    - Install pip-tools and compile/install dependencies:
      ```bash
      pip install pip-tools
      pip-compile
      pip install -r requirements.txt
      ```


## Usage

After completing the installation process, you can now use the transcriber:

  ```python
  import asyncio

  from whisper_realtime_transcriber.InputStreamGenerator import InputStreamGenerator
  from whisper_realtime_transcriber.WhisperModel import WhisperModel
  from whisper_realtime_transcriber.RealtimeTranscriber import RealtimeTranscriber

  inputstream_generator = InputStreamGenerator()
  asr_model = WhisperModel(inputstream_generator)

  transcriber = RealtimeTranscriber(inputstream_generator, asr_model)

  asyncio.run(transcriber.start_event_loop())
  ```

Feel free to reach out if you encounter any issues or have questions!

## How it works

- The transcriber consists of two modules: a [Inputstream Generator](./transcriber/InputStreamGenerator.py) and a [Whisper Model](./transcriber/whisper_models/base.py).
- The implementation of the Inputstream Generator is based on this [implemantation](https://github.com/tobiashuttinger/openai-whisper-realtime).
- The Inputstream Generator reads the microphone input and passes it to the Whisper Model. The Whisper Model then generates the transcription.
- This is happening in an async event loop so that the Whsiper Model can continuously generate transcriptions from the provided audio input, generated and processed by the Inputstream Generator.
- On a machine with a 12GB Nvidia RTX 3060 the [distilled large-v3](https://github.com/huggingface/distil-whisper) model runs at a realtime-factor of about 0.4, this means 10s of audio input get transcribed in 4s - the longer the input the bigger is the realtime-factor.

## ToDos

- Add functionality to transcribe and actually do something with the outputs.
- Add functionality to transcribe from audio files.
- Get somehow rid of the hallucinations of the whisper models when no voice is active.