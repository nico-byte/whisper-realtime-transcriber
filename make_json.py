"""
# Don't start the script unless you want to generate new jsons!
# This Script is responsible for preparing the jsons to be used in the model_eval_script
# If you want to test out the model evaluation script you first need to record this sentences onto your device
# Let the script generate the jsons and look at how the files that you must record are pathed and named
# Make sure to provide a speaker name or list 
"""

speakers = ["speaker1",]

import json
import os

ground_truth = {
    'wetter': [
        "Südlich der Alpen bis Italien runter haben wir momentan kein großes Vergnügen in Sachen Sonnenschein auch der Balkan ist davon betroffen und genervt sind auch viele Urlauber beispielsweise auf Mallorca, wobei es hier jetzt in der nächsten Zeit besser werden wird aber gerade rum um die Adria bleibt es erstmal wechselhaft.",
        "Und ein Teil der feuchten Luft schafft immer den Sprung über den Alpenhaupt sowie der Schauer und Gewitter wird es in den kommenden Tagen bis in den Süd-Osten schaffen, immerhin dass es punktuell mal regen gibt.",
        "Tiefblauer Himmel mit dem Blick über Leipzig aber die kalten Nächte werden immer wieder Bodenfrost und teilweise Luftfrost und dass Anfang Junis und dass sogar mehrere Nächte hintereinander",
        "Geht es los mit der Hitze, und zwar punktuell am Freitag da hätten wir am ober und Teile des hoch Rheins am Mittelrhein im Rhein Main Gebiet zwei bis vier Tage in folge mit plus 30 Grad und in Teilen von Nord Rhein Westfalen und den Niederlanden ein bis 3 Tage mit dieser heißen Temperatur.",
        "Am Montag drohen in Deutschland heftige Gewitter, wenn man ins Zentrum einer Gewitterwolke gerät, kann es stellenweise zumindest kleinräumig Hagelsturm und Starkregen geben.",
        "Der gesamte Westen und der gesamte Norden und auch weite Teile des Osten sind vollkommen Unwetter oder Gewitterfrei.",
        "Im Mittelmeerraum ist es instabil dort können sich heftige Gewitter bilden und bisschen, was von der Gewitterluft schwappt, jetzt eben auch in den Süd-Osten Deutschlands.",
        "Der Start in den Montag ist im Süd-Westen eher angenehm temperiert, jedoch wird es im Osten kühler und zwar nur zwei Grad, örtlich kann es leichten Bodenfrost geben, aber das ist nicht sicher.",
        "Die Sonne wird 13 bis 16 stunden scheinen, es gibt einige hochnebelartige Wolkenfelder, die sich im laufe des Tages auflösen und paar dichtere Wolken im Süd-osten Bayerns mit teilweise regen und es kann zu riesigen quillwolken kommen",
        "Wie wird das Wetter in Norwegen, wenn ich dort Urlaub machen möchte?",
        "Wie ist das aktuelle Wetter in Istanbul?",
        "Wird es heute Abend noch in Nürnberg regnen?"
    ],
    'uhrzeit': [
        "Ich würde gerne heute Mittag gegen 12 oder 13 Uhr, was Essen gehen, willst du mit?",
        "Gehen wir heute noch Feiern oder willst du das ganze lieber verschieben, da du ja eig. nicht mehr nach 22 Uhr rausgehen wolltest.",
        "Wir haben erst 10:45 jetzt können wir uns schön noch 45 min Elektrotechnik anhören.",
        "Wollen wir noch kurz eine Kleinigkeit Essen gehen, bevor wir heute Abend ins gym gehen, denn ich habe das letzte mal um 14 Uhr was gegessen, dass war vor 5 stunden.",
        "Um 15 uhr spiele ich Fußball mit meinen Freunden im Park.",
        "Morgens um 8:30 treffe ich mich mit meiner besten Freundin zum Frühstück.",
        "Um 19 Uhr gehe ich ins Kino, um den neuesten Blockbuster mit meinen Freunden zu sehen.",
        "Nach der Schule um 16:45 gehe ich zum Fußballtraining.",
        "Am Samstag um 12:15 treffe ich mich mit meinen Freunden in einem Cafe.",
        "Um 20:30 gehe ich mit meiner Familie zum Abendessen in ein Restaurant.",
        "Jeden Mittwoch um 17 uhr habe ich Gitarrenunterricht.",
        "Um 14:45 Uhr treffe ich mich mit meinen Freunden im Park zum Picknick.",
        "Am Sonntag um 10 uhr gehe ich mit meinem Hund spazieren.",
        "Um 18:30 spiele ich Videospiele mit meinen Freunden online.",
        "Nach der Arbeit um 19:15 treffe ich mich mit meinen Kollegen in der Bar.",
        "Am Freitag um 13:30 gehe ich zum Schwimmtraining.",
        "Um 11:45 gehe ich mit meiner Schwester einkaufen.",
        "Um 21 uhr treffe ich mich mit meinen Freunden in einer Bowlinghalle.",
        "Am Wochenende um 9:30 mache ich Yoga mit meiner besten Freundin.",
        "Morgen um 11 uhr werde ich mit meinen Freunden eine Fahrradtour machen, bei der wir verschiedene Sehenswürdigkeiten in der Stadt besichtigen und anschließend ein Picknick im Park veranstalten.",
        "Um 16:30 habe ich eine Verabredung mit meiner besten Freundin, bei der wir in einem gemütlichen Café sitzen, Kaffee trinken und über die neuesten Bücher diskutieren werden.",
        "Am Samstag um 14 uhr werde ich mit meiner Fußballmannschaft ein wichtiges Turnier spielen, bei dem wir gegen den lokalen Rivalen antreten und unser Bestes geben werden, um den Sieg zu erringen.",
        "Um 19:45 treffe ich mich mit meinen Freunden in einem italienischen Restaurant, um gemeinsam zu Abend zu essen und anschließend einen lustigen Spieleabend zu veranstalten, bei dem wir Karten- und Brettspiele spielen werden."
        "Wann gehen wir heute Abend in den Club?",
        "Wann wollen wir uns zum Lernen treffen?",
        "Um wieviel Uhr musst du in der Früh aufstehen?",
        "Wollten wir uns nicht um 15 Uhr am Fitnessstudio treffen?"
    ],
    'datum': [
        "Am dritten Januar habe ich mich mit meinen Freunden im Fitnessstudio getroffen, um das neue Jahr mit einem gemeinsamen Workout zu beginnen.",
        "Nächsten Montag, am fünfzehnten Februar, werde ich mich nach der Arbeit mit meiner besten Freundin im Fitnessstudio verabreden, um an einem Zumba-Kurs teilzunehmen.",
        "Am fünften März werde ich mich mit meinen Trainingspartnern im Fitnessstudio treffen, um gemeinsam an einem intensiven Gewichthebungsprogramm zu arbeiten.",
        "Anfang April plane ich ein Outdoor-Training mit meinen Freunden, bei dem wir gemeinsam joggen und verschiedene Übungen im Park machen werden.",
        "Am zwanzigsten Mai treffen wir uns im Fitnessstudio, um unsere Fortschritte zu messen und an neuen Trainingsplänen zu arbeiten.",
        "Im Juni werde ich mich zweimal pro Woche mit meiner Laufgruppe treffen, um uns auf einen bevorstehenden 10-Kilometer-Lauf vorzubereiten.",
        "Am zwölften Juli werde ich mich mit meinem Trainingspartner im Fitnessstudio treffen, um eine intensive HIIT-Trainingseinheit durchzuführen.",
        "Ende August plane ich einen gemeinsamen Ausflug mit meinen Freunden zum Klettern in einer Indoor-Kletterhalle.",
        "Am fünften September werde ich mich mit meinen Fitnessstudio-Freunden treffen, um an einem Yoga-Retreat-Wochenende teilzunehmen.",
        "Am achtzehnten Oktober werde ich meine Freunde zu einem Basketballspiel im Fitnessstudio herausfordern und ein Turnier veranstalten.",
        "Im November treffen wir uns regelmäßig im Fitnessstudio, um uns auf den bevorstehenden Winterlauf vorzubereiten.",
        "Am zweiten Dezember werde ich mit meinen Trainingspartnern im Fitnessstudio an einem Gruppen-Fitnesskurs teilnehmen.",
        "Anfang des nächsten Jahres plane ich ein gemeinsames Wochenend-Radfahr-Abenteuer mit meinen Freunden.",
        "Im Februar werde ich mich mit meiner besten Freundin im Fitnessstudio treffen, um neue Workouts auszuprobieren und uns gegenseitig zu motivieren.",
        "Am zwanzigsten März werde ich eine Fitnessparty organisieren, bei der ich meine Freunde zum Tanzen und Spaßhaben ins Fitnessstudio einlade.",
        "Kannst du mir das heutige Datum nennen?",
        "Welches Datum haben wir übermorgen?",
        "Welches Datum haben wir in 7 Tagen?"
    ],
    'witze': [
        "Warum haben Seemänner keine E-Mail-Adressen? Weil sie lieber Wellen machen!",
        "Was macht eine Biene auf dem Fitnessgerät? Sie summt!",
        "Warum hat der Frosch keinen Job? Weil er immer nur kroch!",
        "Was ist grün und singt? Ein Kühlschrank, der 'Oper' macht!",
        "Warum nehmen Skelette immer ihr Essen mit ins Bett? Weil sie Angst vor dem Knochenbrecher haben!",
        "Warum hat der Mathematiklehrer Probleme mit seinem Garten? Weil er seine Wurzeln nicht loswerden kann!",
        "Was sagt ein Stift zum Radiergummi? 'Du bist so radikal!''",
        "Warum hat der Komiker immer einen Regenschirm dabei? Für den perfekten Schauer!",
        "Warum hat der Tennisspieler nie Geld dabei? Weil er seine Bälle verloren hat!",
        "Was sagt ein Zahnarzt zu einem Fußballer? 'Sie haben ein Tor, aber keine Zähne!'",
        "Was ist gelb und kann nicht schwimmen? Ein Bagger. Findest du den Witz lustig? Nicht? Der Baggerfahrer auch nicht.",
        "Kannst du mir einen Witz sagen",
        "Erzähl mir einen Witz",
        "Mach mal einen Witz"
    ]
}

# function to add to JSON
# Load your existing data
metadata = {'dataset': []}
for speaker in speakers:
    speaker_data = {}
    # Loop over each category and its list in ground_truth
    for category, texts in ground_truth.items():
        speaker_data[category] = []
        # Reset the counter at the beginning of each new category
        for i, text in enumerate(texts, start=1):
            # Preprocess the text
            # processed_text = preprocess_text(text)
            new_entry = {
                "filepath": f"./data/{speaker}/" + str(i).zfill(4) + "_" + category + f"_{speaker}.wav",
                "ground_truth": text,
                "transcriptions": {},
                "runtime_values": {},
            }
            # Find the corresponding category in the existing data and append the new entry
            speaker_data[category].append(new_entry)
    metadata['dataset'].append(speaker_data)

if not os.path.exists(f"./data/{speaker}/"):
    os.makedirs(f"./data/{speaker}/")

# Write the updated data back into the file
for speaker in speakers:
    with open(f'./data/{speaker}/metadata.json', 'a', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
