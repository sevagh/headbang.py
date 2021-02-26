import time
import pyaudio
import time
import numpy
from madmom.io.audio import write_wave_file
import librosa
import sys
import argparse
from headbang.util import load_wav, overlay_clicks


class Player:
    def __init__(self, audio_file, chunk_size=1024, sr=44100):
        self.frame_count = chunk_size

        self.half_sr = int(sr)

        self.audio, self.sample_rate = librosa.load(
            audio_file, sr=sr, dtype=numpy.float32, mono=False
        )
        self.stereo = False

        if len(self.audio.shape) > 1 and self.audio.shape[1] == 2:
            self.stereo = True

        self.cycle_count = 0
        self.pa = pyaudio.PyAudio()

    def pyaudio_callback(self, in_data, frame_count, time_info, status):
        audio_size = numpy.shape(self.audio)[1]

        if frame_count * self.cycle_count > audio_size:
            return (None, pyaudio.paComplete)
        elif frame_count * (self.cycle_count + 1) > audio_size:
            frames_left = audio_size - frame_count * self.cycle_count
        else:
            frames_left = frame_count

        data = self.audio[
            :,
            frame_count * self.cycle_count : frame_count * self.cycle_count
            + frames_left,
        ]

        if self.stereo:
            data = data.reshape((2 * data.shape[1],))

        out_data = data.astype(numpy.float32).tobytes()
        self.cycle_count += 1
        return (out_data, pyaudio.paContinue)

    def start_player(self):
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.half_sr,
            output=True,
            input=False,
            stream_callback=self.pyaudio_callback,
            frames_per_buffer=self.frame_count,
        )

        # Start the stream
        self.stream.start_stream()

    def processing(self):
        return self.stream.is_active()

    def terminate_processing(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        self.cycle_count = 0


def main():
    parser = argparse.ArgumentParser(
        prog="annotate_beats.py",
        description="Beat tracking - human annotator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("beat_wav_out", help="output beat wav file")

    args = parser.parse_args()

    player = Player(args.wav_in, chunk_size=1024)

    print("When audio plays, hit any key to mark a beat")
    print("When audio ends, hit another key to exit the input loop")
    input("... hit ret to start ...")

    start = time.time()
    player.start_player()

    beats = []

    while True and player.processing():
        input()
        if not player.processing():
            break
        beats.append(time.time() - start)

    beats = numpy.asarray(beats)

    # subtract my reaction delay
    beats -= 0.3

    player.terminate_processing()

    print("annotated beat locations: {0}".format(beats))
    x = load_wav(args.wav_in, stereo=True)
    beat_waveform = overlay_clicks(x, beats)

    print("Writing outputs with clicks to {0}".format(args.beat_wav_out))
    write_wave_file(beat_waveform, args.beat_wav_out, sample_rate=44100)


if __name__ == "__main__":
    main()
