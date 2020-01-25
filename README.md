# DiphoneSynth
A basic object-oriented text-to-speech program using diphone speech synthesis.

### Usage

Requires PyAudio, Numpy, and NLTK.

Note for Windows users: PyAudio must be installed via pipwin (i.e `pip install pipwin` followed by `pipwin install pyaudio`)

Important: unpack the diphones.7z archive such that the diphones folder is on the same level as the .py files!

Run diphone_synth.py in the command line with the following arguments:

- *"Your phrase"*, as the text you want to synthesise. Default: None
- *--play* or *-p*, to play the generated waveform. Default: True
- *--crossfade* or *-c*, to enable crossfading (TD-PSOLA) of diphones for smoother-sounding output. Default: False
- *--volume* or *-v*, to specify the volume of the audio output (range 1 to 100). Default: 100
- *--save* or *-s* followed by *filename.wav*, to save the output to a new file, path relative. Default: None
- *--help* or *-h* to open the help menu with these instructions.


In theory, you can use a different diphone database. You will have to update the global variable SAMPLE_RATE to match that of your wav files. The filename conventions will also have to be the same.
