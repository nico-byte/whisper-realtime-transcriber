import asyncio
import sys

from whisper_realtime_transcriber.InputStreamGenerator import InputStreamGenerator
from whisper_realtime_transcriber.WhisperModel import WhisperModel
from whisper_realtime_transcriber.RealtimeTranscriber import RealtimeTranscriber


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
