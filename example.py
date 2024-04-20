import asyncio
import sys

from transcriber.whisper_models.finetuned import FinetunedWhisper
from transcriber.whisper_models.distilled import DistilWhisper
from transcriber.whisper_models.stock import StockWhisper
from transcriber.InputStreamGenerator import InputStreamGenerator


def main():
    transcriber_conf = {
        'model_params': {
            'model_id': None,
            'model_size': 'small',
            'device': 'cpu',
            'language': 'en'
        },
        'generator_params': {
            'samplerate': 16000,
            'blocksize': 4000,
            'adjustment_time': 5,
            'memory_safe': True
        }
    }
    
    # Load inputstream_generator
    inputstream_generator = InputStreamGenerator(**transcriber_conf['generator_params'])
    
    # Load model
    # when using the FinetunedWhisper class one can specify a different whisper model from huggingface
    # in the transcriber_conf
    # model_size becomes obsolete then
    asr_model = DistilWhisper(inputstream_generator=inputstream_generator, **transcriber_conf['model_params'])
    
    asyncio.run(start(inputstream_generator, asr_model))
        
async def start(inputstream_generator, asr_model):
    # Create a transcribe and inputstream task
    inputstream_task = asyncio.create_task(inputstream_generator.process_audio())
    transcribe_task = asyncio.create_task(asr_model.run_inference())
    
    # Execute the tasks and catch exception
    try:
        await asyncio.gather(inputstream_task, transcribe_task)
    except asyncio.CancelledError:
        print("\nTranscribe task cancelled.")


if __name__ == '__main__':
    try:
        print("Activating wire...")
        main()
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')