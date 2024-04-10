import random
import aiohttp
import wikipedia
import pytz

from datetime import datetime as dtime
from aiogoogletrans import Translator
from model_eval_script import preprocess_text

# prepare the paths to the generated audio data
generated_joke = "./generated_joke.wav"
generated_time = "./generated_time.wav"
generated_datetime = "./generated_datetime.wav"
generated_weather = "./generated_weather.wav"
generated_failure = "./generated_failure.wav"


async def jokes():
    """
    # This function is responsible for choosing a rendom joke and converting it to audio
    """

    joke = ["Zwei Freundinnen treffen sich: 'Wie viel wiegst du?"
            "Keine Antwort. Komm schon, wenigstens die ersten 3 Zahlen…",
            "Verkehrskontrolle. Der Polizist: Haben Sie etwas getrunken? Autofahrer: Nein."
            "Polizist: Sollten Sie aber! Mindestens 2 Liter am Tag.",
            "Freitagabend. Schatz, sollen wir uns ein schönes Wochenende machen? Klar!"
            "„Klasse, dann bis Montag!"]

    # generate the audio
    print(random.choice(joke))

async def current_time():
    """
    This function is responsible for getting the current time in germany and converting it to audio
    """

    # get the current time
    tz = pytz.timezone('Europe/Berlin')
    dt_now = dtime.now(tz)
    dt = dt_now.strftime("%H:%M")
    hour = dt[0:2]
    minute = dt[3:5]

    if minute[0] == "0":
        minute = minute[1]

    if hour[0] == "0":
        hour = hour[1]

    text = f"Es ist {hour} Uhr {minute}."
    # print(text)

    print(text)


async def current_datetime():
    """
        This function is responsible for getting the current date in germany and converting it to audio
        """

    # get the current date
    tz = pytz.timezone('Europe/Berlin')
    dt_now = dtime.now().astimezone(tz)
    dt = dt_now.strftime("%Y:%d:%m")
    year = dt[0:4]
    day = dt[5:7]
    month = dt[8:10]

    if day[0] == "0":
        day = day[1]

    if month[0] == "0":
        month = month[1]

    text = f"Es ist der {day + 'te'} {month + 'te'} {year}."

    print(text)


async def get_weather(transcript):
    """
    This function is responsible for getting the current weather in one or several cities and converting it to audio
    """
    cities = []

    # check if cities are present in the transcription
    for text in transcript:
        try:
            summary = str(wikipedia.summary(text))
            if 'city' in summary:
                cities.append(text)
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for {text}: {e.options}")
        except wikipedia.exceptions.PageError:
            print(f"Page for {text} not found in wikipedia")

    # get the current weather in the cities
    for city in cities:
        async with aiohttp.ClientSession() as session:
            # initialize the translator
            translator = Translator()
            async with session.get(f'https://wttr.in/{city}?format=%C|%t|%w') as response:
                # translate and preprocess the text
                text = await response.text()
                data = text.split("|")
                processed_data = preprocess_text(data[1])

                translation = await translator.translate(data[0], dest="de")
                weather_conditions = translation.text
                processed_weather_conditions = preprocess_text(weather_conditions)

                text = f'In {city} ist es {processed_weather_conditions}. Es herrscht eine Durchschnittstemperatur von {processed_data[0:2]} Grad Celsius mit einer Windgeschwindigkeit von {data[2]}'

                print(text)


async def failure():
    text = "Das kann ich leider nicht."
    print(text)