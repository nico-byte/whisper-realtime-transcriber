import asyncio
import sys
import argparse
import yaml

from typing import Dict
from transcriber.whisper_models.finetuned import FinetunedWhisper
from transcriber.whisper_models.distilled import DistilWhisper
from transcriber.whisper_models.stock import StockWhisper
from transcriber.InputStreamGenerator import InputStreamGenerator

def check_config(args):
    # Set default values in case config file is borken/nonexistent
    defaults = {
        'model_params': {
            'backend': 'stock',
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
    
    try:
        with open(args.transcriber_conf) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(config, Dict):
            return defaults

    except FileNotFoundError:
        print(f"Could not find config file in {args.transcriber_conf}.")
        return defaults
    
    return config

def main(transcriber_conf):
    # Load inputstream_generator
    inputstream_generator = InputStreamGenerator(**transcriber_conf['generator_params'])
    
    # Load model based on desired backend
    backend = transcriber_conf['model_params']['backend']
    if backend == "finetuned":
        asr_model = FinetunedWhisper(inputstream_generator=inputstream_generator, **transcriber_conf['model_params'])
    elif backend == "stock":
        asr_model = StockWhisper(inputstream_generator=inputstream_generator, **transcriber_conf['model_params'])
    else:
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
    # Add parser and arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcriber_conf', type=str, default="./transcriber_config.yaml", help='Config file for the transcriber (default: ./transcriber_config.yaml).')
    
    args = parser.parse_args()
    transcriber_conf = check_config(args)
    print(transcriber_conf)
    try:
        print("Activating wire...")
        main(transcriber_conf)
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')