import asyncio
import sys
import os

from whisper_realtime_transcriber.InputStreamGenerator import InputStreamGenerator, GeneratorArguments
from whisper_realtime_transcriber.WhisperModel import WhisperModel, ModelArguments
from whisper_realtime_transcriber.RealtimeTranscriber import RealtimeTranscriber


async def _print_transcriptions(transcriptions: list) -> None:
    """
    Prints the model transcription.
    """
    output = [transcription for transcription in transcriptions if transcription != [""]]

    os.system("cls") if os.name == "nt" else os.system("clear")

    for transcription in output:
        words = transcription.split(" ")
        line_count = 0
        split_input = ""
        for word in words:
            line_count += 1
            line_count += len(word)
            if line_count > os.get_terminal_size().columns:
                split_input += "\n"
                line_count = len(word) + 1
                split_input += word
                split_input += " "
            else:
                split_input += word
                split_input += " "

        print(split_input)
    print("", end="", flush=True)


def main():
    model_args = ModelArguments(device="mps")
    generator_args = GeneratorArguments()

    transcriber = RealtimeTranscriber(model_args, generator_args, func=_print_transcriptions)

    asyncio.run(transcriber.execute_event_loop())


if __name__ == "__main__":
    try:
        print("Activating wire...")
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
