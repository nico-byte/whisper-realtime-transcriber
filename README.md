# Whisper Realtime Transcriber

## Overview

This repository contains the source code for a realtime transcriber for various whisper models, published on huggingface.

## Prerequisites

Before you begin, make sure you have the following installed on your machine:

- [Python + Conda](https://www.anaconda.com/download)

## Installation Process

Follow the steps below to set up the project on your local machine:

1. **Clone the Project:**
   - Clone this repository to your local machine using `git`:
     ```bash
     git clone https://github.com/nico-byte/whisper-realtime-transcriber
     ```

2. **Create conda environment:**
   - Create a `conda` environment named `whisper-realtime`:
     ```bash
     conda create --name whisper-realtime python=3.10.12
     ```
   - Now activate the `conda` environment:
     ```bash
     conda activate whisper-realtime
     ``` 

3. **Install dependencies:**
   - Install all dependencies via `pip`:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

After completing the installation process, you can now use the transcriber:

- First let's look at the [transcriber_config.yaml](./transcriber_config.yaml) file. This file contains all the necessary information for the generator and models of the transcriber to run.
  ```yaml
  model_params:
    backend: 'distilled'  # 'stock', 'finetuned', 'distilled'
    model_id: ''          # will only be used when choosing 'finetuned' as backend
    model_size: 'large'   # 'small', 'medium', 'large' - obsolete when using a custom model_id
    device: 'cuda'        # 'cuda', 'mps', 'cpu'
    language: 'en'        # language is only used for tokenizing the output of the models, the models detect the language automatically - the distilled models only work with english
  generator_params:
    samplerate: 16000     # samplerate of the audio input, anything you like
    blocksize: 4000       # the size of the blocks that are processed by the generator at once, anything you like - 4000 is the best value i found
    adjustment_time: 5    # duration in seconds for adjusting the silence threshold
  ```

- Of course another file can be used to configure the transcriber. The default one is [transcriber_config.yaml](./transcriber_config.yaml).

- One can now use the transcriber like this:
  ```bash
  python -m transcriber --transcriber_conf=transcriber_config.yaml
  ```

- The `trascriber_conf` argument can be ignored unless another one is used!

- The [example.py](./example.py) is another example of how to use the transcriber.

Feel free to reach out if you encounter any issues or have questions!