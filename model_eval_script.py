"""
# Don't start the script unless you want to generate new metrics!
# existing metrics will be extended/overwritten
# If you want to generate new metrics, delete the existing metrics.json, provdide the speaker names, 
# make sure you record the samples provided in the make_json.py script and run the script again.
"""


import torch
import whisper  # is installed as openai-whisper in the requirements.txt
import librosa
import regex as re
import json
import string
import numpy as np
import time

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, Wav2Vec2ForCTC, Wav2Vec2Processor
from pytorch_installation_val import set_device
from num2words import num2words


device = set_device()

speakers = ["speaker1",]


def preprocess_text(text):
    """
    # This function is responsible to preprocess our text data to ensure consistency in the data
    """

    # Identify and convert times
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

    return text


def load_data(json_file, category_name, c):
    """
    # This function is responsible for loading the jsons and loading the paths to the audio files
    """

    # load the json
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # create empty list
    audios = []
    # append audio paths to the empty list
    for x, _ in enumerate(data["dataset"][c][category_name]):
        audios.append(data["dataset"][c][category_name][x]['filepath'])
    print(f'Audiofile Arrays: {audios}')
    return audios


def wer(ref, hyp, debug=True):
    """
    # This function is responsible for computing the metrics
    """

    lines = None
    r = ref.split()
    h = hyp.split()
    # costs will hold the costs, like in the Levenshtein distance algorithm
    costs = [[0 for _ in range(len(h) + 1)] for _ in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for _ in range(len(h) + 1)] for _ in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    wer_result = round((numSub + numDel + numIns) / float(len(r)), 3)
    return {'WER': wer_result, 'numCor': numCor, 'numSub': numSub, 'numIns': numIns, 'numDel': numDel,
            "numCount": len(r)}


def speech_file_to_array_fn(file_path):
    # convert the content of the audio files to time series
    speech_array, _ = librosa.load(file_path, sr=16_000, mono=True)
    return speech_array


def write_json(data, model_type, model_identifiers, audio, category_name, audio_files, c):
    """
    # This function is responsible for adding the generated entries to the jsons
    """

    # if the data is empty, we can't edit it
    if data is None:
        raise ValueError("Data is None")
    # load data
    with open(data, 'r', encoding='utf-8') as file:
        json_file = json.load(file)

    # main loop for processing the audio and adding the transcriptions to the json
    for y, audio_data in enumerate(audio):
        # convert the time series to a waveform
        waveform = torch.from_numpy(np.array(audio_data))
        print(f'Processing audio {y+1}/{len(audio)}')

        # append a transcriptions dict if it's not already there
        if "transcriptions" not in json_file['dataset'][c][category_name][y]:
            json_file['dataset'][c][category_name][y]["transcriptions"] = {}

        # append a runtime_values dict if it's not already there
        if "runtime_values" not in json_file['dataset'][c][category_name][y]:
            json_file['dataset'][c][category_name][y]["runtime_values"] = {}

        print(f'Current audio path: {audio_files[y]}')
        print(f'Current waveform: {waveform}')

        # generate new entry in data
        entry = run_models(json_file, model_type, model_identifiers, waveform, audio_files[y], y, category_name, c)

        # write new entry to the json
        with open(data, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)


def run_models(data, model_type, model_identifiers, waveform, audio_file, y, category_name, c):
    """
    # This function is responsible for running the models and returning the generated transcriptions/runtime_values
    """

    if model_type == "ft_whisper":

        for model_id in model_identifiers:
            # if an entry already exists skip it
            if model_id in data['dataset'][c][category_name][y]["transcriptions"]:
                print(f'{model_id} is already present.')
                continue
            print(f'Model ID {model_id} running')
            start_time = time.time()

            # set the model and processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
            processor = AutoProcessor.from_pretrained(model_id, language="german", task="transcribe")
            model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")

            # generate usable inputs
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features
            input_features = input_features.to(device)

            # Transcribe the audio using the pre-trained model
            generated_ids = model.generate(inputs=input_features, max_new_tokens=225)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            clean_transcription = preprocess_text(transcription)

            # set the entries
            data['dataset'][c][category_name][y]["transcriptions"][model_id] = clean_transcription
            data['dataset'][c][category_name][y]["runtime_values"][model_id] = round(time.time() - start_time, 3)

    if model_type == "vanilla_whisper":
        for model_id in model_identifiers:
            # if an entry already exists skip it
            if model_id in data['dataset'][c][category_name][y]["transcriptions"]:
                print(f'{model_id} is already present.')
                continue
            print(f'Model ID {model_id} running')
            start_time = time.time()

            # set the model
            model = whisper.load_model(model_id)

            # Transcribe the audio using the  model
            transcription = model.transcribe(audio_file)
            clean_transcription = preprocess_text(transcription['text'])

            # set the entries
            data['dataset'][c][category_name][y]["transcriptions"][model_id] = clean_transcription
            data['dataset'][c][category_name][y]["runtime_values"][model_id] = round(time.time() - start_time, 3)

    if model_type == "wav2vec":
        for model_id in model_identifiers:
            # if an entry already exists skip it
            if model_id in data['dataset'][c][category_name][y]["transcriptions"]:
                print(f'{model_id} is already present.')
                continue
            print(f'Model ID {model_id} running')
            start_time = time.time()

            # set the model and processor
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            model = Wav2Vec2ForCTC.from_pretrained(model_id)

            # generate usable inputs
            inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt", padding=True)

            with torch.no_grad():
                logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

            # Transcribe the audio using the pre-trained model
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentence = processor.batch_decode(predicted_ids)
            clean_transcription = preprocess_text(predicted_sentence[0])

            # set the entries
            data['dataset'][c][category_name][y]["transcriptions"][model_id] = clean_transcription
            data['dataset'][c][category_name][y]["runtime_values"][model_id] = round(time.time() - start_time, 3)

    if model_type == "mms":
        for model_id in model_identifiers:
            # if an entry already exists skip it
            if model_id in data['dataset'][c][category_name][y]["transcriptions"]:
                print(f'{model_id} is already present.')
                continue
            print(f'Model ID {model_id} running')
            start_time = time.time()

            # set the model and processor
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            model = Wav2Vec2ForCTC.from_pretrained(model_id)

            processor.tokenizer.set_target_lang("deu")
            model.load_adapter("deu")

            # generate usable inputs
            inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = model(**inputs).logits

            # Transcribe the audio using the model
            ids = torch.argmax(outputs, dim=-1)[0]
            transcription = processor.decode(ids)
            clean_transcription = preprocess_text(transcription)

            # set the entries
            data['dataset'][c][category_name][y]["transcriptions"][model_id] = clean_transcription
            data['dataset'][c][category_name][y]["runtime_values"][model_id] = round(time.time() - start_time, 3)

    return data


def load_metrics_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            metrics_data = json.load(file)
            return metrics_data.copy()
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filename}.")
        return None


def main():
    """
    # This function is responsible for evaluating the models
    """

    # create dict with the model identifiers
    model_identifiers = {
        "ft_whisper": [
            "bofenghuang/whisper-small-cv11-german",
            "bofenghuang/whisper-medium-cv11-german",
            "bofenghuang/whisper-large-v2-cv11-german"
            ],
        "vanilla_whisper": [
            "tiny",
            "base",
            "small",
            "medium",
            "large"
        ],
        "wav2vec": [
            'jonatasgrosman/wav2vec2-large-xlsr-53-german',
            'jonatasgrosman/wav2vec2-xls-r-1b-german'
        ],
        "mms": [
            'facebook/mms-1b-all',
            'facebook/mms-1b-fl102'
        ]
    }

    # Load the metrics data
    metrics_file_path = './metrics_template.json'  # Adjust the path if your file is in a different directory
    metrics_file = load_metrics_from_file(metrics_file_path)

    if metrics_file is not None:
        print("Metrics data loaded successfully.")
        # You can now work with the metrics_data dictionary in your script
        # For example, print it out
        # print(json.dumps(metrics_file, indent=4))
    else:
        print("Failed to load metrics file.")

    # create lists containing the speakers and categories
    # the lists are updated to avoid overwriting our existing data
    categories = ["wetter", "uhrzeit", "datum", "witze"]

    """
    # 1. main loop
    # 2. iterating over the speakers, generate the path to the according json
    # 3. iterate over the categories to generate the time series for the audio files
    # 4. iterate over the model_identifiers to process the audio data/generate the entries in the jsons
    """
    for s in speakers:
        json_file = './data/' + s + '/metadata.json'
        for c, category in enumerate(categories):
            audios = load_data(json_file, category, c)
            speech_arrays = []
            for x in range(len(audios)):
                speech_arrays.append(speech_file_to_array_fn(audios[x]))
            for k in model_identifiers.keys():
                print(f'Modeltype {k} running...')
                model_id_list = [d for d in model_identifiers[k]]
                print(f'The following models are running now: {model_id_list}')
                print(f'DEBUG - Current Audio Paths: {audios}')
                write_json(json_file, k, model_id_list, speech_arrays, category, audios, c)
                
    """
    # 1. iterate over the speakers and get the path to the according json
    # 2. iterate over the categories
    # 3. iterate over the model identifiers and create a list for the according model type
    # 4. iterate over the models
    # 5. iterate over the ground truth and transcription entries in the json and generate the raw metrics according to our data
    # 6. write the metrics to the metrics.json
    """
    for speaker in speakers:
        with open('./data/' + speaker + '/metadata.json', 'r', encoding='utf-8') as file:
            transcriptions_file = json.load(file)
        for c, category in enumerate(categories):
            for model_type in model_identifiers.keys():
                model_id_list = [d for d in model_identifiers[model_type]]
                for m, model in enumerate(model_id_list):
                    for k in range(len(transcriptions_file['dataset'][c][category])):
                        results = wer(transcriptions_file['dataset'][c][category][k]['ground_truth'],
                                      transcriptions_file['dataset'][c][category][k]['transcriptions'][model])

                        metrics_file['metrics'][0][model_type][m]['all_errors'].append(results['numSub'] + results['numIns'] + results['numDel'])
                        metrics_file['metrics'][0][model_type][m]['all_wer'].append(results['WER'])
                        metrics_file['metrics'][0][model_type][m]['all_token_count'].append(results['numCount'])
                        
                        with open('./metrics.json', 'w', encoding='utf-8') as f:
                            json.dump(metrics_file, f, indent=2, ensure_ascii=False)

    # runtime_values
    for speaker in speakers:
        with open('./data/' + speaker + '/metadata.json', 'r', encoding='utf-8') as file:
            transcriptions_file = json.load(file)
        for c, category in enumerate(categories):
            for model_type in model_identifiers.keys():
                model_id_list = [d for d in model_identifiers[model_type]]
                for m, model in enumerate(model_id_list):
                    for k in range(len(transcriptions_file['dataset'][c][category])):
                        runtime_value = transcriptions_file['dataset'][c][category][k]['runtime_values'][model]

                        if "runtime_values" not in metrics_file['metrics'][0][model_type][m]:
                            metrics_file['metrics'][0][model_type][m]["runtime_values"] = []
                        
                        metrics_file['metrics'][0][model_type][m]['runtime_values'].append(runtime_value)
                        
                        with open('./metrics.json', 'w', encoding='utf-8') as f:
                            json.dump(metrics_file, f, indent=2, ensure_ascii=False)

    for model_type in model_identifiers.keys():
        model_id_list = [d for d in model_identifiers[model_type]]
        for m, model in enumerate(model_id_list):
            runtime_values = metrics_file['metrics'][0][model_type][m]['runtime_values']
            len_runtime_values = len(metrics_file['metrics'][0][model_type][m]['runtime_values'])

            if "avg_runtime" not in metrics_file['metrics'][0][model_type][m]:
                metrics_file['metrics'][0][model_type][m]["avg_runtime"] = {}

            if len_runtime_values != 0:
                avg_runtime = round(sum(runtime_values) / len_runtime_values, 2)
            else:
                avg_runtime = 0.0

            metrics_file['metrics'][0][model_type][m]['avg_runtime'] = avg_runtime

            with open('./metrics.json', 'w', encoding='utf-8') as f:
                json.dump(metrics_file, f, indent=2, ensure_ascii=False)

    """
    # 1. iterate over the model identifiers and create a list for the according model type
    # 2. iterate over the models
    # 3. iterate over the metrics and compute the avg metrics like avg_wer, avg_token_count ...
    # 4. write the metrics to the metrics.json
    """
    for model_type in model_identifiers.keys():
        model_id_list = [d for d in model_identifiers[model_type]]
        for m, model in enumerate(model_id_list):
            all_errors = metrics_file['metrics'][0][model_type][m]['all_errors']
            len_all_errors = len(metrics_file['metrics'][0][model_type][m]['all_errors'])
            all_token_count = metrics_file['metrics'][0][model_type][m]['all_token_count']
            len_all_token_count = len(metrics_file['metrics'][0][model_type][m]['all_token_count'])

            if len_all_errors != 0:
                avg_errors = sum(all_errors) / len_all_errors
            else:
                avg_errors = 0.0

            metrics_file['metrics'][0][model_type][m]['avg_errors'] = avg_errors

            if len_all_token_count != 0:
                avg_token_count = sum(all_token_count) / len_all_token_count
            else:
                avg_token_count = 0

            metrics_file['metrics'][0][model_type][m]['avg_token_count'] = avg_token_count

            if sum(all_token_count) != 0:
                avg_wer = round(sum(all_errors) / sum(all_token_count), 3)
            else:
                avg_wer = 0.0

            metrics_file['metrics'][0][model_type][m]['avg_wer'] = avg_wer

            with open('./metrics.json', 'a', encoding='utf-8') as f:
                json.dump(metrics_file, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
