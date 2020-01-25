# DiphoneSynth
A basic object-oriented text-to-speech program using diphone speech synthesis.

### Usage

Requires PyAudio, Pylab, Numpy, and NLTK.

Note for Windows users: PyAudio must be installed via pipwin (i.e `pip install pipwin` followed by `pipwin install pyaudio`)

Run diphone_synth.py in the command line with the following arguments:



The 'diphones' folder should be in the same directory as diphone_synth.py and interface_audio.py. 

In theory, you can use a different diphone database. You will have to update the global variable SAMPLE_RATE to match that of your wav files. The filename conventions will also have to be the same.
