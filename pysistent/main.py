import asyncio
import sys

from Inference import Inference
from InputStreamGenerator import InputStreamGenerator
from play_audio import play_audio
from bark import preload_models


async def main():
    # await asyncio.to_thread(preload_models())
    
    inputstream_generator = await init_inputstream_generator()
    print("Successfully loaded inputsream generator.")
    print("Loading models...")
    
    wakeword_model = Inference(model_task="transcribe", model_type="vanilla", model_size="base", device="cuda")
    wakeword_model.load()
    print("Successfully loaded wakeword model.")
    
    asr_model = Inference(model_task="transcribe", model_type="pretrained", model_size="medium", device="cuda")
    asr_model.load()
    print("Successfully loaded asr model.")
    
    tts_model = Inference(model_task="tts", device="cuda")
    tts_model.load()
    print("Successfully loaded tts model.")
    
    while True:
        print("Listening...")
        audio_data = await inputstream_generator.record(2)
        print(audio_data)
        
        wakeword_model.run(audio_data=audio_data)
        print(wakeword_model.transcript)
        print(wakeword_model.processed_transcript)
        
        if "thorsten" in wakeword_model.processed_transcript or "torsten" in wakeword_model.processed_transcript:
            print("Wakeword detected.")
            audio_data = await inputstream_generator.record(7)
            
            asr_model.run(audio_data=audio_data)
            print("Generated transcript: " + asr_model.transcript)
            
            tts_model.run(text=asr_model.transcript)
            print("Generated audio file: " + tts_model.path_to_file)
            
            await asyncio.to_thread(play_audio, tts_model.path_to_file)
            

async def init_inputstream_generator():
    inputstream_generator = await InputStreamGenerator(samplerate=16000, blocksize=24678, silence_ratio=1000, adjustment_time=5)
    # await inputstream_generator.set_silence_threshold()
    return inputstream_generator
            
         

if __name__ == '__main__':
    try:
        print("Activating wire...")
        preload_models()
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')