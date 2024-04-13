import asyncio
import sys

from pysistent.LiveAudioTranscriber import LiveAudioTranscriber


async def main():        
    asr_model = await LiveAudioTranscriber(model_type="vanilla", model_size="small", device="cuda")
    load_model = asyncio.to_thread(asr_model.load)
    await load_model
    
    await asr_model.set_silence_threshold()
    
    transcribe_task = asyncio.create_task(asr_model.transcribe())
    
    try:
        await transcribe_task
    except asyncio.CancelledError:
        print("Transcribe task cancelled.")


if __name__ == '__main__':
    try:
        print("Activating wire...")
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')