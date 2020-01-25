import pyaudio as pa
import numpy as np
import wave

# define default audio format values
CHUNK = 256
FORMAT = pa.paInt16
CHANNELS = 1
SAMPLE_RATE = 48000
MAX_AMP = 2**15  # for rescale


class Audio(pa.PyAudio):
    """
    Audio object with functions to play .wav data as audio, save data to new .wav, and change volume
    based on a factor from 0 to 1.
    Also contains support functions.
    Primarily, use .play(),
                   .save(filename), where filename is a string ending in '.wav'
                   .rescale(factor), where factor is a float or int between 0 and 1
    """

    def __init__(self, channels=1, rate=SAMPLE_RATE, chunk=CHUNK, format=FORMAT):
        pa.PyAudio.__init__(self)  # initialise parent class
        self.channels = channels
        self.sample_rate = rate
        self.chunk = chunk
        self.format = format
        self.nptype = self.get_nptype(format)  # figure out the current array type
        self.data = np.array([], dtype=self.nptype)  # set the current data to an empty array of the correct type
        self.ostream = None  # a closed output stream
        self.chunk_index = 0  # a counter for referencing the data in chunks

    def __del__(self):
        self.terminate()

    def add_chunk(self):
        """Add a chunk of data into the current output stream."""
        slice_from = self.chunk_index * self.chunk
        slice_to = slice_from + self.chunk - 1
        # numpy does not raise indexerror when slicing out of bounds, so check it here
        if slice_to > self.data.shape[0]:
            raise IndexError
        array = self.data[slice_from:slice_to]
        self.ostream.write(array.tostring())
        self.chunk_index += 1

    def open_output_stream(self):
        """Open an output stream with current format attributes."""
        self.ostream = self.open(format=self.format, channels=self.channels, rate=self.sample_rate, output=True)
        self.chunk_index = 0

    def close_output_stream(self):
        """Close the output stream, e.g when finished playing audio."""
        self.ostream.close()
        self.ostream = None

    def load(self, path):
        """Load audio data from a given .wav file."""
        wave_file = wave.open(path, "rb")
        self.format = self.get_format_from_width(wave_file.getsampwidth())  # get format info from header
        self.nptype = self.get_nptype(self.format)  # get nptype from format
        self.channels = wave_file.getnchannels()  # get number of channels from header
        self.sample_rate = wave_file.getframerate()  # get sample rate from header
        self.data = np.array([], dtype=self.nptype)  # make an empty ndarray to write to
        raw_data = wave_file.readframes(self.chunk)  # read a data chunk from the file
        while raw_data:  # while there is still data in the file...
            array = np.fromstring(raw_data, dtype=self.nptype)  # make an ndarray from the raw data
            self.data = np.append(self.data, array)  # add it to the full data array
            raw_data = wave_file.readframes(self.chunk)  # start reading the next chunk
        wave_file.close()

    def play(self):
        """Play given audio data."""
        self.open_output_stream()
        print("Playing...")
        while True:
            try:    
                self.add_chunk()
            except IndexError:
                break  # if we run out of data to output, break out of the loop
        print("Stopped playing")
        self.close_output_stream()

    def save(self, path):
        """Save audio data to a .wav file."""
        raw_data = self.data.tostring()  # cast data to string for writing
        wav_file = wave.open(path, 'wb')
        wav_file.setnchannels(self.channels)  # set channel info for header
        wav_file.setsampwidth(self.get_sample_size(self.format))  # set format info
        wav_file.setframerate(self.sample_rate)   # set sample rate info
        wav_file.writeframes(raw_data)  # write the data
        wav_file.close()  # close the file

    def get_nptype(self, type):
        """Convert common pyaudio types to numpy types."""
        if type == pa.paInt24:
            return np.int24
        elif type == pa.paInt16:
            return np.int16
        elif type == pa.paInt8:
            return np.int8

    def rescale(self, factor):
        """Change audio volume based on a given rescale factor between 0 and 1, independent of system volume."""
        if 0 <= factor <= 1:
            peak = np.max(np.abs(self.data))  # define peak to prevent clipping
            rescale_factor = factor * MAX_AMP / peak  # multiply every data point in the array by rescale factor
            self.data = (self.data * rescale_factor).astype(self.nptype)
        else:
            print("Expected a scaling factor between 0 and 1")
