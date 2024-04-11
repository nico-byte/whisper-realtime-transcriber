"""
# First year project of:
# Nico Fuchs
# 
# 
"""


import whisper  # installed as openai-whisper via requirements.txt
import asyncio
import sys
import numpy as np
import sounddevice as sd
import torch

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pysistent.text_processing import preprocess_text
from interactions import jokes, current_time, current_datetime, get_weather, failure
from pytorch_installation_val import set_device
from timeit import default_timer as timer

print(sys.executable)

device = set_device()

# Load models
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained("bofenghuang/whisper-medium-cv11-german").to(device)
asr_processor = AutoProcessor.from_pretrained("bofenghuang/whisper-medium-cv11-german", language="german", task="transcribe")
wakeword_model = whisper.load_model("base")
asr_model.config.forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="de", task="transcribe")

# SETTINGS
BLOCKSIZE = 24678
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds.
SILENCE_THRESHOLD = int()
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO = 1000
# number of samples in one buffer that are allowed to be higher than threshold
SAMPLERATE = 16000
# number of samples per second
ADJUSTMENT_TIME = 5
# timespan for setting the SILENCE_THRESHOLD
wakeword_global_ndarray = None
asr_global_ndarray = None

DEBUG = False

# Set magic words
activation_word = ["hey", "thorsten"]
joke_keywords = ["witze", "witz", "witzen"]
time_keywords = ["zeit", "spät", "uhrzeit"]
weather_keywords = ["wetter"]
datetime_keywords = ["datum", "tag", "jahr", "monat"]

async def inputstream_generator():
    """
    # Generator that yields blocks of input data as NumPy arrays
    """
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(in_data, _, __, state):
        loop.call_soon_threadsafe(q_in.put_nowait, (in_data.copy(), state))

    stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def wakeword_processing():
    """
    # This function is responsible for processing the input stream and detecting the wakewords
    """

    global wakeword_global_ndarray

    async for wakeword_indata, _ in inputstream_generator():
        wakeword_indata_flattened = abs(wakeword_indata.flatten())

        # discard buffers that contain mostly silence
        if np.nonzero(wakeword_indata_flattened > SILENCE_THRESHOLD).size < SILENCE_RATIO:
            continue

        if wakeword_global_ndarray is not None:
            wakeword_global_ndarray = np.concatenate((wakeword_global_ndarray, wakeword_indata), dtype='int16')
        else:
            wakeword_global_ndarray = wakeword_indata

        # concatenate buffers if the end of the current buffer is not silent and if the chunksize is under 5
        if np.average((wakeword_indata_flattened[-100:-1])) > SILENCE_THRESHOLD / 15 and wakeword_indata_flattened.size / 16000 < 2:
            continue
        else:
            start = timer()
            print("Processing Wakeword...")
            wakeword_local_ndarray = wakeword_global_ndarray.copy()
            wakeword_global_ndarray = None
            wakeword_processed_transcript = await run_wakeword(wakeword_local_ndarray)
            if DEBUG:
                print(f'Result: {wakeword_processed_transcript}\nElapsed processing time: {timer() - start} seconds\n')
            if activation_word[0] and activation_word[1] in wakeword_processed_transcript:
                print("Wakeword activation successfull!")
                return True
            else:
                print('Wakeword not detected!')
                print(f'This was transcribed: {wakeword_processed_transcript}')
                

async def run_wakeword(wakeword_ndarray):
    print("Processing Wakeword...")
    wakeword_indata_transformed = wakeword_ndarray.flatten().astype(np.float32) / 32768.0
    # transcribe the time series
    wakeword_result = wakeword_model.transcribe(wakeword_indata_transformed, language="de")
    wakeword_processed_transcript = await asyncio.to_thread(preprocess_text, wakeword_result["text"])
    
    return wakeword_processed_transcript


async def asr(wakeword):
    """
    # This function is responsible for processing the input stream and handling what interactions should be triggered
    # It only gets triggered if the wakeword model detected the wakewords
    """

    global asr_global_ndarray

    while(wakeword):
        async for asr_indata, _ in inputstream_generator():
            asr_indata_flattened = abs(asr_indata.flatten())

            # discard buffers that contain mostly silence
            if np.nonzero(asr_indata_flattened > SILENCE_THRESHOLD).size < SILENCE_RATIO:
                continue

            if asr_global_ndarray is not None:
                asr_global_ndarray = np.concatenate((asr_global_ndarray, asr_indata), dtype='int16')
            else:
                asr_global_ndarray = asr_indata

            # concatenate buffers if the end of the current buffer is not silent and if the chunksize is under 5
            if np.average((asr_indata_flattened[-100:-1])) > SILENCE_THRESHOLD / 15 and asr_indata_flattened.size / 16000 < 5:
                continue
            else:
                start = timer()
                print("Processing ...")
                asr_local_ndarray = asr_global_ndarray.copy()
                asr_global_ndarray = None
                asr_indata_transformed = asr_local_ndarray.flatten().astype(np.float32) / 32768.0

                asr_processed_transcript = await run_inference(asr_indata_transformed)
                
                if DEBUG:
                    print(f'Result: {asr_processed_transcript}\nElapsed processing time: {timer() - start} seconds\n')
                # decide what interaction is wanted
                if any(x in asr_processed_transcript for x in joke_keywords):
                    if DEBUG:
                        print("Jokes")
                    return await jokes()
                if any(x in asr_processed_transcript for x in time_keywords):
                    if DEBUG:
                        print("Time")
                    return await current_time()
                if any(x in asr_processed_transcript for x in datetime_keywords):
                    if DEBUG:
                        print("Datetime")
                    return await current_datetime()
                if any(x in asr_processed_transcript for x in weather_keywords):
                    if DEBUG:
                        print("Wetter")
                    return await get_weather(asr_processed_transcript)
                else:
                    return await failure()
                
                
async def run_inference(source):
    """
    # This function is responsible for starting the inference
    """
    # Transform the waveform into the required format
    waveform = torch.from_numpy(source)
    inputs = asr_processor(waveform, sampling_rate=SAMPLERATE, return_tensors="pt")
    input_features = inputs.input_features
    input_features = input_features.to(device)
    # Transcribe the audio using the pre-trained model
    generated_ids = asr_model.generate(inputs=input_features, max_new_tokens=225)
    asr_result = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    asr_processed_transcript = await asyncio.to_thread(preprocess_text, asr_result)
    
    return asr_processed_transcript


async def set_silence_threshold():
    """
    # This function is responsible for setting the SILENCE_THRESHOLD to the environment loudness
    """
    global SILENCE_THRESHOLD
    
    print('\nActivating wire ...\n')
    # Set the desired input-/output devices
    print(f'Available devices (you can set your desired devices as default if not already):\n{sd.query_devices()}')
    index_of_input_device = None    # system default
    index_of_output_device = None   # system default
    sd.default.device = [index_of_input_device, index_of_output_device]
    print(f'If you changed the devices, here you can validate it:\n{sd.query_devices()}')
    
    blocks_processed = 0
    loudness_values = []

    async for indata, _ in inputstream_generator():
        blocks_processed += 1
        indata_flattened = abs(indata.flatten())

        # Compute loudness over first few seconds to adjust silence threshold
        loudness_values.append(np.mean(indata_flattened))

        # Stop recording after ADJUSTMENT_TIME seconds
        if blocks_processed >= ADJUSTMENT_TIME * SAMPLERATE / BLOCKSIZE:
            SILENCE_THRESHOLD = int(np.mean(loudness_values) * SILENCE_RATIO)
            break
    
    print(f'\nSet SILENCE_THRESHOLD to {SILENCE_THRESHOLD}\n')


async def main():
    """
    # This function is responsible for executing the set_silence_threshold and starting the main event loop
    """
    # start the main loop
    # asyncio.create_task(wakeword_processing())
    await set_silence_threshold()
    
    while True:
        wakeword = await wakeword_processing()
        await asr(wakeword)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')

# https://github.com/tobiashuttinger/openai-whisper-realtime/blob/main/openai-whisper-realtime.py
