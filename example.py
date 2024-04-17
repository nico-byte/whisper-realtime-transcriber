import asyncio
import sys

from transcriber.whisper_models.finetuned import FinetunedWhisper
from transcriber.whisper_models.distilled import DistilWhisper
from transcriber.whisper_models.stock import StockWhisper
from transcriber import InputStreamGenerator


async def main():        
    # Load inputstream_generator
    inputstream_generator = await InputStreamGenerator()
    
    # Load model config
    asr_model = await DistilWhisper(inputstream_generator=inputstream_generator, model_size="large", language="en", device="cuda")
    await asr_model.load()
    
    # Create a transcribe task
    inputstream_task = asyncio.create_task(inputstream_generator.process_audio())
    transcribe_task = asyncio.create_task(asr_model.run_inference())
    
    # Execute the task and catch exception
    try:
        await asyncio.gather(inputstream_task, transcribe_task)
    except asyncio.CancelledError:
        print("Transcribe task cancelled.")


if __name__ == '__main__':
    try:
        print("Activating wire...")
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')