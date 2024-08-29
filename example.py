import asyncio
import sys

from transcriber.InputStreamGenerator import InputStreamGenerator
from transcriber.WhisperModel import WhisperModel
from transcriber.RealtimeTranscriber import RealtimeTranscriber


def main():
    inputstream_generator = InputStreamGenerator()
    asr_model = WhisperModel(inputstream_generator)
    
    transcriber = RealtimeTranscriber(inputstream_generator, asr_model)

    asyncio.run(transcriber.start_event_loop())


if __name__ == "__main__":
    try:
        print("Activating wire...")
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
