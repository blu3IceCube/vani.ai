import tkinter as tk
import sounddevice as sd
import numpy as np
import threading
import os
from scipy.io import wavfile
from faster_whisper import WhisperModel

class AudioRecorder:
    def __init__(self, master):
        self.master = master
        master.title("Audio Recorder")

        self.button = tk.Button(master, text="Start Recording", command=self.toggle_recording)
        self.button.pack(pady=20)

        self.is_recording = False
        self.frames = []
        self.sample_rate = 16000

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.button.config(text="Stop Recording")
        self.frames = []
        threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        self.is_recording = False
        self.button.config(text="Start Recording")

    def record_audio(self):
        with sd.InputStream(samplerate=self.sample_rate, channels=1) as stream:
            while self.is_recording:
                audio_data, overflowed = stream.read(self.sample_rate)
                self.frames.append(audio_data)
        
        self.save_audio()

    def save_audio(self):
        audio_data = np.concatenate(self.frames, axis=0)
        wavfile.write("audio.wav", self.sample_rate, audio_data)
        
        # Convert wav to mp3 using ffmpeg
        # os.system("ffmpeg -i audio.wav -acodec libmp3lame -b:a 128k audio.mp3")
        # os.remove("audio.wav")  # Remove the temporary wav file
        
        self.transcribe_audio()

    def transcribe_audio(self):
        print("Transcribing audio...")
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        segments, info = model.transcribe("audio.wav", beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        transcription = ""
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            transcription += segment.text + " "

        # Display transcription in a new window
        transcription_window = tk.Toplevel(self.master)
        transcription_window.title("Transcription")
        tk.Label(transcription_window, text=transcription, wraplength=400).pack(padx=10, pady=10)

root = tk.Tk()
app = AudioRecorder(root)
root.mainloop()