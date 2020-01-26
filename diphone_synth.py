#!/usr/bin/env python

# DiphoneSynth
# A basic object oriented text-to-speech program using diphone speech synthesis.
# hypnaceae on github
# License: GNU GPL v3

import os
import audio_interface
import argparse
import nltk
from nltk.corpus import cmudict
import re
import numpy as np

parser = argparse.ArgumentParser(
    description='A basic object oriented text-to-speech program using diphone speech synthesis.')
parser.add_argument('--diphones', default="./diphones", help="Relative path to folder containing diphone .wavs")
parser.add_argument('--play', '-p', action="store_true", default=True, help="Play the processed audio data")
parser.add_argument('--save', '-s', action="store", dest="outfile", type=str, help="Save the audio output to a file",
                    default=None)
parser.add_argument('phrase', nargs=1, help="The phrase to be synthesised")
parser.add_argument('--crossfade', '-c', action="store_true", default=False,
                    help="Enable smoother concatenation by cross-fading between diphone units, i.e TD-PSOLA")
parser.add_argument('--volume', '-v', default=100, type=int,
                    help="Integer between 0 and 100, representing final output volume")
#parser.add_argument('--help', '-h', help="Shows help menu.")
args = parser.parse_args()

# regex to match DD/MM/YY or DD/MM/YYYY, can also match . or - separators
dmy_reg = re.compile(r"""
       (?P<day>^0?[1-9]|[12][0-9]|3[01])       # group as "day" all matches between 1 and 31 at the start of the string
       [/.-]                                   # match / . or - (don't group)
       (?P<month>[1-9]|1[012])                 # group as "month" all matches between 1 and 12
       [/.-]                                   
       (?P<year>(?:\d{4}|\d{2})$)              # group as "year" any 2 or 4 digit number at end of the string
       """, re.VERBOSE)
# regex to match DD/MM, can also match . or - separators
dm_reg = re.compile(r"""
       (?P<day>^0?[1-9]|[12][0-9]|3[01])       # group as "day" all matches between 1 and 31 at the start of the string
       [/.-]                                   # match / . or - (don't group)
       (?P<month>[1-9]|1[012])$                # group as "month" all matches between 1 and 12 at the end of the string
       """, re.VERBOSE)

SAMPLE_RATE = 16000  # the sample rate of .wav files in ./diphones.


class Utterance:
    """
    The Utterance object: the frontend of the TTS system.
    Here we tokenise the user's input, normalise dates into
    speakable words, and build a sequence of diphones.

    Much extension opportunity for this class. Can normalise
    many more types of numbers: e.g monetary values, numbers alone,
    time, percentages etc. As is, the synthesiser will fail to process
    these tokens.
    """
    def __init__(self, phrase):
        self.final_tokenisation = list()
        self.diphone_sequence = list()
        self.utterance_phrase = phrase      # args.phrase[0]

    def tokenise(self):
        """
        Simple first tokenisation. Using NLTK's standard tokeniser, build a list of the words in the utterance.
        Any tokens that match the date regexes are passed to normalise_dates to be expanded into words themselves,
        and are returned to this method to be added to the final list of tokens.
        """

        phrase_tokenised = nltk.word_tokenize(self.utterance_phrase)

        try:
            for token in phrase_tokenised:
                if re.match(dmy_reg, token) is not None \
                        or re.match(dm_reg, token) is not None:      # if the token corresponds to regex above,
                    expanded_date = self.normalise_dates(token)      # pass the current token to date normalisation
                    self.final_tokenisation += expanded_date.copy()  # and add it to the final list
                else:
                    self.final_tokenisation.append(token)            # if there is no regex match, use original token

            print("Tokens:", self.final_tokenisation)

        except Exception as e:
            print(e)  # since most errors here will be caused by incorrect/missing args, just print any error.
            print("Please make sure you are using arguments as specified in --help or the readme.")

    def normalise_dates(self, token):
        """      ---      DATE EXPANSION      ---
        This module takes a token in the format DD/MM, DD/MM/YY, or DD/MM/YYYY. It returns this as a list of strings
        such as "June Twenty Eighth, Nineteen Fourteen".

        This function could be extended as "normalise_numbers", handling dates and other values in phrase input such
        as integers (up to trillion?), monetary values, times, mathematical expressions etc, because CMU dictionary
        does not contain non-word characters.
        """

        # set up tuples with names. naturally, integer input will be used as index to find the corresponding word.
        # index 0 is set as some placeholder value in day_names and month_names
        day_names = ("Zeroth", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth",
                     "Tenth", "Eleventh,", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth",
                     "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth", "Thirtieth", "Twenty", "Thirty")

        month_names = ("N/Anuary", "January", "February", "March", "April", "May", "June", "July",
                       "August", "September", "October", "November", "December")

        under_twenty = ('Oh', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven',
                    'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen')

        tens = ('Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety')

        result_date = []

        # begin tedious expansion of date tokens
        date_matcher = dmy_reg.match(token)
        if date_matcher:
            if 0 < int(date_matcher.group(1)) < 20:  # handle days
                result_date.append(day_names[int(date_matcher.group(1))])
            elif int(date_matcher.group(1)) == 20:
                result_date.append(day_names[20])
            elif 20 < int(date_matcher.group(1)) < 30:
                result_date.append(day_names[22])
                result_date.append(day_names[int(date_matcher.group(1)) % 10])
            elif int(date_matcher.group(1)) == 30:
                result_date.append(day_names[21])
            else:
                result_date.append(day_names[23])
                result_date.append(day_names[1])

            result_date.append(month_names[int(date_matcher.group(2))])  # handle month
            result_date.append(",")

            # with the following code, only years from 1000 to 2099 are supported
            if 1000 <= int(date_matcher.group(3)) <= 1999:
                result_date.append(under_twenty[int(date_matcher.group(3))[:1]])  # take only the millennium and century
            elif 2000 <= int(date_matcher.group(3)) <= 2099:
                result_date.append(tens[0])

            year = int(date_matcher.group(3)) % 100  # get last two digits of year

            if 1 <= year < 10:  # handle first decade separately
                result_date.append(under_twenty[0])
                result_date.append(under_twenty[year])
            elif 10 <= year < 20:  # handle teens
                result_date.append(under_twenty[year])
            elif 20 <= year <= 99:  # handle rest of the decades...
                ten, under_ten = divmod(year, 10)  # get quotient and remainder to use
                result_date.append(tens[ten - 2])  # as indices for getting name from 20 to 99
                result_date.append(under_twenty[under_ten]) 
            else:
                print("Invalid year.")  # this shouldn't happen

            # clean up trailing "Oh"
            if result_date[-1] == "Oh":
                result_date.pop(-1)

        date_matcher = dm_reg.match(token)  # now, for handling DD/MM cases. This is largely repeated from above,
        if date_matcher:                    # so check if we might optimise this a bit.

            if int(date_matcher.group(1)) < 20:  # handle days
                result_date.append(day_names[int(date_matcher.group(1))])
            elif int(date_matcher.group(1)) == 20:
                result_date.append(day_names[20])
            elif 20 < int(date_matcher.group(1)) < 30:
                result_date.append(day_names[22])
                result_date.append(day_names[int(date_matcher.group(1)) % 10])
            elif int(date_matcher.group(1)) == 30:
                result_date.append(day_names[21])
            else:
                result_date.append(day_names[23])
                result_date.append(day_names[1])

            result_date.append(month_names[int(date_matcher.group(2))])  # handle month

        return result_date

    def get_phone_seq(self):
        """---      LETTER TO SOUND      ---
        This module creates a diphone sequence from the tokenised phrase by first looking up each word token in the CMU
        phones dictionary, and then joining pairs of these into diphones. It also replaces punctuation with appropriate
        lengths of silence.
        """

        print("Processing diphones...")

        # define punctuation types; all, short, and long. first tuple may seem redundant, but it makes later code
        # a bit more efficient, and also enables proper functioning of punctuation-to-silence replacement.
        punctuation = ("\\", "/", ",", ":", ";", "—", "(", ")", "[", "]", "{", "}", "\"", "?", "!", ".", "...")
        punctuation_short = ("\\", "/", ",", ":", ";", "—", "(", ")", "[", "]", "{", "}", "\"")
        punctuation_long = ("?", "!", ".", "...")

        phone_dict = cmudict.dict()
        phones_list = ['PAU']  # start with a phrase-initial pause
        for token in self.final_tokenisation:
            if token in punctuation:
                phones_list.append(token)  # punctuation has no phone, so add as-is.
            else:
                try:
                    phones_list += phone_dict.get(token.lower())[:1].copy()[0]  # add the token's phones to the list
                except TypeError:
                    print("Error:", token, "is not in the CMU dictionary. Try changing it to a word.")
                    quit(0)  # catch exception in case of unrecognised token (including numerals)

        phones_list.append('PAU')  # end with a phrase-final pause

        # now format the raw phone list into a list of diphones we can work with
        # re.sub is used to remove unwanted numbers from the phone name.
        # basically, the following algorithm takes the name of the current phone and adds the name of the next phone,
        # with a hyphen, corresponding to the filenames in ./diphones.
        diphone_sequence = []
        for i, elem in enumerate(phones_list):
            nextelem = phones_list[(i + 1) % len(phones_list)]  # define next element in the list
            previouselem = phones_list[(i - 1)]  # define previous element in the list

            if previouselem in punctuation:  # put a pause after punctuation, for smooth transition from it
                diphone_sequence.append("PAU-" + re.sub(r'\d+', '', elem))
            if nextelem in punctuation:  # put a pause before punctuation, for smooth transition into it
                diphone_sequence.append(re.sub(r'\d+', '', elem) + "-PAU")
            elif elem not in punctuation and nextelem not in punctuation:  # otherwise just make the diphone
                diphone_sequence.append(re.sub(r'\d+', '', elem) + "-" + re.sub(r'\d+', '', nextelem))

            if elem in punctuation:  # now replace punctuation with markers for silences of appropriate length
                if elem in punctuation_long:
                    diphone_sequence.append("400ms-silence")
                elif elem in punctuation_short:
                    diphone_sequence.append("200ms-silence")

        return diphone_sequence


class Synth:
    """
    Get the output of Utterance.get_phone_seq() (a list of diphones), get the corresponding filenames of each diphone, 
    put the appropriate length of silences, make a temporary audio processing object to help in concatenating all the
    audio data of each diphone.
    """

    def __init__(self):
        self.diphones = {}

    def get_wavs(self, wav_folder=args.diphones):
        """Convert the joined diphone list into a list of .wav filenames that can later be loaded as audio."""

        # make a set of existing diphone names to check against, so we don't get errors down the line
        wav_list = set([])
        for root, dirs, files in os.walk(wav_folder, topdown=False):
            for file in files:
                wav_list.add(file)

        phone_sequence = utt.get_phone_seq()  # get the sequence of phones in user's phrase

        # build the actual filenames based on the diphone list from get_phone_seq
        wavs_for_concatenation = []
        for diphone in phone_sequence:  # still need to include silences, since they will be replaced by zeroes later
            if diphone == "200ms-silence" or diphone == "400ms-silence":
                wavs_for_concatenation.append(diphone)
            elif (diphone.lower() + ".wav") in wav_list:
                wavs_for_concatenation.append(args.diphones + "/" + diphone.lower() + ".wav")  # make diphone filenames
            else:
                print("Diphone", diphone, "not found in", args.diphones, ". Skipping...")  # skip missing diphones
                continue

        return wavs_for_concatenation

    def make_and_concatenate_chunks(self):
        """
        First insert the actual silences (i.e zeroes), and load the corresponding audio data for each diphone file.
        Includes the option to crossfade each file. Finally, return a fully-formed ndarray that can be fed into
        our audio interface.
        """

        temp_audio = audio_interface.Audio(rate=SAMPLE_RATE)

        # define silences as an ndarray of some length of zeroes
        two_hundred_ms_silence = np.zeros(int(SAMPLE_RATE * 0.2), temp_audio.nptype)
        four_hundred_ms_silence = np.zeros(int(SAMPLE_RATE * 0.4), temp_audio.nptype)

        # get the list of files we need to concatenate
        wav_filenames_and_silences = self.get_wavs()

        # make a list of actual audio chunks, replacing the silence marker with the corresponding zeroes
        chunks_out_list = []
        for chunk in wav_filenames_and_silences:
            if chunk == "200ms-silence":
                chunks_out_list.append(two_hundred_ms_silence)
            elif chunk == "400ms-silence":
                chunks_out_list.append(four_hundred_ms_silence)
            else:
                temp_audio.load(chunk)
                chunks_out_list.append(temp_audio.data)

        # TD-PSOLA: taper head and tail of chunks towards 0 over 10ms, overlap by 10ms
        if args.crossfade:
            print("Crossfading...")
            millisec = int(SAMPLE_RATE/100)  # equivalent to 10ms worth of samples at the current rate
            # build a list of factors, from 0 to 1, which we will use to taper the volume of each chunk
            crossfade_factor = [(round(0 + (i * (1 / millisec)), 3)) for i in range(0, millisec)]

            # first, taper the chunks
            tapered_chunks_out_list = []
            for chunk in chunks_out_list:
                # make tapered first and last 10ms of the chunk
                head_taper = np.multiply(crossfade_factor, chunk[:millisec])
                tail_taper = np.multiply(crossfade_factor, chunk[-millisec:])
                chunk[:millisec] = head_taper   # replace the first 10ms with our new tapered data
                chunk[-millisec:] = tail_taper  # replace the last 10ms
                tapered_chunks_out_list.append(chunk)

            # next, overlap the tapered chunks
            overlapped_chunks = []
            deletion_index = [i for i in range(0, millisec)]  # define a list of indices to delete when overlapping
            for i, chunk in enumerate(tapered_chunks_out_list):
                next_chunk = tapered_chunks_out_list[(i + 1) % len(tapered_chunks_out_list)]
                # overlap tail of current chunk with head of next chunk
                overlapped_tail = np.add(chunk[-millisec:], next_chunk[:millisec])
                chunk[-millisec:] = overlapped_tail
                # 'move' the next chunk by deleting the first 10ms, which are now in the last 10ms of the current chunk.
                shortened_next_chunk = np.delete(next_chunk, deletion_index)
                tapered_chunks_out_list[(i + 1) % len(tapered_chunks_out_list)] = shortened_next_chunk
                overlapped_chunks.append(chunk)

            concatenated_chunks = np.concatenate(overlapped_chunks)

        else:  # if crossfade option not used
            concatenated_chunks = np.concatenate(chunks_out_list)

        return concatenated_chunks


if __name__ == "__main__":

    utt = Utterance(args.phrase[0])  # instantiate utterance object with user's phrase
    utt.tokenise()  # do the tokenisation
    synth = Synth()  # instantiate synthesis object
    final_output = audio_interface.Audio(rate=SAMPLE_RATE)  # instantiate output object to take our final synth data
    final_output.data = synth.make_and_concatenate_chunks()  # give it the final audio data array

    # handle volume:
    if args.volume:
        if 100 >= args.volume >= 0:
            final_output.rescale(args.volume / 100)     # convert user input to range 0,1
        else:
            print("--volume/-v expected one argument between 0 and 100.")
            quit(0)

    # play!
    if args.play:
        final_output.play()

    # save the file:
    if args.outfile:
        try:
            print("Saving:", args.outfile)
            final_output.save(args.outfile)
        except FileNotFoundError:
            print("Could not find the directory to save audio output to. Please check the filename string supplied to "
                  "--save/-s.")
            quit(0)
