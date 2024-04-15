import asyncio
import sys

from transcriber.pretrained import PretrainedWhisper
from transcriber import LiveAudioTranscriber


async def main():        
    # Load model config
    asr_model = await PretrainedWhisper(model_size="small", device="cuda")
    await asr_model.load()
    
    # Load transcriber
    transcriber = await LiveAudioTranscriber()
    
    # Create a transcribe task
    transcribe_task = asyncio.create_task(transcriber.transcribe(model=asr_model, loop_forever=True))
    
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