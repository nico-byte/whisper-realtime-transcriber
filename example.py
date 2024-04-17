import asyncio
import sys

from transcriber.whisper_models.finetuned import FinetunedWhisper
from transcriber.whisper_models.distilled import DistilWhisper
from transcriber.whisper_models.stock import StockWhisper
from transcriber import InputStreamGenerator


async def main():        
    # Load inputstream_generator
    inputstream_generator = await InputStreamGenerator()
    
    # Load model
    # when using the FinetunedWhisper class one can specify a different whisper model from huggingface
    # like this:
    # model_id = "bofenghuang/whisper-large-cv11-german",
    # asr_model = await FinetunedWhisper(inputstream_generator=inputstream_generator, model_id=model_id)
    # model_size becomes obsolete then
    asr_model = await FinetunedWhisper(inputstream_generator=inputstream_generator, model_size="large", device="cuda")
    
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
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')