import asyncio
import sys
import argparse

from transcriber.whisper_models.finetuned import FinetunedWhisper
from transcriber.whisper_models.distilled import DistilWhisper
from transcriber.whisper_models.stock import StockWhisper
from transcriber.InputStreamGenerator import InputStreamGenerator


def add_args(parser):
    parser.add_argument('--backend', type=str, default="distilled", choices=["distilled", "finetuned", "stock"],help='Backend to be used for Whisper processing (default: distilled).')
    parser.add_argument('--model_size', type=str, default='large', choices=["small", "medium", "large"],help="Size of the Whisper model to be used (default: large).")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "mps", "cpu"],help='Device to be used for Whisper processing (default: cuda).')

async def main(args):        
    # Load inputstream_generator
    inputstream_generator = await InputStreamGenerator()
    
    # Load model
    backend = args.backend
    if backend == "finetuned_whisper":
        asr_model = await FinetunedWhisper(inputstream_generator=inputstream_generator, model_size=args.model_size, device=args.device)
    elif backend == "stock_whisper":
        asr_model = await StockWhisper(inputstream_generator=inputstream_generator, model_size=args.model_size, device=args.device)
    else:
        asr_model = await DistilWhisper(inputstream_generator=inputstream_generator, model_size=args.model_size, device=args.device)
    
    # Create a transcribe and inputstream task
    inputstream_task = asyncio.create_task(inputstream_generator.process_audio())
    transcribe_task = asyncio.create_task(asr_model.run_inference())
    
    # Execute the tasks and catch exception
    try:
        await asyncio.gather(inputstream_task, transcribe_task)
    except asyncio.CancelledError:
        print("\nTranscribe task cancelled.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    try:
        print("Activating wire...")
        asyncio.run(main(args))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')