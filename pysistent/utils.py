import sounddevice as sd
import soundfile as sf
import re
import string
import nltk

from num2words import num2words
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

def play_audio(path_to_file: str):
    fs1, x = sf.read(path_to_file, dtype='float32')
    sd.play(fs1, x)
    sd.wait()
    sd.stop()


def preprocess_text(text):
    """
    # This function is responsible for preprocessing the text, to ensure consistency in our data
    """

    # Identify and convert times
    nltk.download('stopwords', quiet=True)
    text = text.lower()
    time_pattern = r"\b\d{1,2}:\d{2}\b"
    times = re.findall(time_pattern, text)
    for time in times:
        hours, minutes = map(int, time.split(":"))
        time_in_words = num2words(hours, lang='de') + " uhr " + num2words(minutes, lang='de')
        text = text.replace(time, time_in_words)

    # Keep German umlauts
    remove_punct_map = {ord(char): None for char in string.punctuation if char not in ['ä', 'ö', 'ü', 'ß']}
    # Convert the text to lowercase, remove punctuation and strip white spaces
    text = text.translate(remove_punct_map).strip()

    # Convert numbers to words
    text = ' '.join(num2words(int(word), lang='de') if word.isdigit() else word for word in text.split())

    tokens = WordPunctTokenizer().tokenize(text)

    # Get the list of german stop words
    german_stop_words = stopwords.words('german')
    # Remove the german stopwords from the list of tokens
    norm_tokens = [x for x in tokens if x not in german_stop_words]

    return norm_tokens
