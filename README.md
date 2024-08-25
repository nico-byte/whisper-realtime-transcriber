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
  - Install all dependencies via `venv`:
    - Make the install script (if needed):
      ```bash
      chmod +x install.sh
      ```
    - Execute the install script:
      ```bash
      ./install.sh
      ```
  - Install all dependencies via [Conda](https://anaconda.org):
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

- First let's look at the [transcriber_config.yaml](./transcriber_config.yaml) file. This file contains all the necessary information for the generator and models of the transcriber to run.
  ```yaml
  model_params:
    backend: 'distilled'  # 'stock', 'finetuned', 'distilled'
    model_id: ''          # will only be used when choosing 'finetuned' as backend
    model_size: 'small'   # 'small', 'medium', 'large' - obsolete when using a custom model_id
    device: 'cpu'         # 'cuda', 'mps', 'cpu'
    punctuate_truecase: False  # whether to punctuate/truecas the model output or not because in partial trnascriptions these might be worng
  generator_params:
    samplerate: 16000     # samplerate of the audio input, anything you like
    blocksize: 4000       # the size of the blocks that are processed by the generator at once, anything you like
                          # -> 4000 is the best value i found
    adjustment_time: 5    # duration in seconds for adjusting the silence threshold
    memory_safe: True     # if True, the generator will discard all buffers that are generated during model inference 
                          # -> set to True if your device is not fast enough to keep up with the generator
  ```

- Of course another file can be used to configure the transcriber. The default one is [transcriber_config.yaml](./transcriber_config.yaml).

- One can now use the transcriber like this:
  ```bash
  python -m transcriber --transcriber_conf=transcriber_config.yaml
  ```

- The `trascriber_conf` argument can be ignored unless another one is used!

- The [example.py](./example.py) is another example of how to use the transcriber.

Feel free to reach out if you encounter any issues or have questions!

## How it works

- The transcriber consists of two modules: a [Inputstream Generator](./transcriber/InputStreamGenerator.py) and a [Whisper Model](./transcriber/whisper_models/WhisperBase.py).
- The implementation of the Inputstream Generator is based on this [implemantation](https://github.com/tobiashuttinger/openai-whisper-realtime).
- The Inputstream Generator reads the microphone input and passes it to the Whisper Model. The Whisper Model then generates the transcription.
- This is happening in an async event loop so that the Whsiper Model can continuously generate transcriptions from the provided audio input, generated and processed by the Inputstream Generator.
- On a machine with a 12GB Nvidia RTX 3060 the [distilled large-v3](https://github.com/huggingface/distil-whisper) model runs at a realtime-factor of about 0.4, this means 10s of audio input get transcribed in 4s, by only utilizing around 1.8GB of VRAM.

## ToDos

- Add webinterface for convenience.
  - I'v tried a few frameworks like Gradio or DearPyGui, but i find it tedious to work with them so this is not a priority.
- Add functionality to transcribe from audio files.
- Get somehow rid of the hallucinations of the whisper models when no voice is active.