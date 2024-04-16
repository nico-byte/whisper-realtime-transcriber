import asyncio
import sys

from transcriber.whisper_models.finetuned import FinetunedWhisper
from transcriber.whisper_models.distilled import DistilWhisper
from transcriber.whisper_models.stock import StockWhisper
from transcriber import LiveAudioTranscriber


async def main():        
    # Load model config
    asr_model = await DistilWhisper(model_size="large", language="en", device="cuda")
    await asr_model.load()
    
    # Load transcriber
    transcriber = await LiveAudioTranscriber()
    
    # Create a transcribe task
    transcribe_task = asyncio.create_task(transcriber.transcribe(model=asr_model, loop_forever=False))
    
    # Execute the task and catch exception
    try:
        transcript, original_tokens, processed_tokens = await transcribe_task
    except asyncio.CancelledError:
        print("Transcribe task cancelled.")

    # Print model output and tokens
    print(f"Transcript: {transcript}")
    print(f"Original tokens: {original_tokens}")
    print(f"Processed tokens: {processed_tokens}")


if __name__ == '__main__':
    try:
        print("Activating wire...")
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')