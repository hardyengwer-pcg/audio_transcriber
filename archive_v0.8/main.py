import os
import sys
import threading
import soundcard as sc
import soundfile as sf
import google.generativeai as genai
import numpy as np
import warnings
from dotenv import load_dotenv, set_key

# Suppress specific soundcard warnings that flood the terminal
try:
    from soundcard.mediafoundation import SoundcardRuntimeWarning
    warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
except ImportError:
    pass

# Load environment variables
load_dotenv()

ENV_PATH = ".env"

def get_api_key():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found.")
        return None
    return api_key

def save_api_key(api_key):
    # Save to .env file
    try:
        if not os.path.exists(ENV_PATH):
            with open(ENV_PATH, "w") as f:
                f.write("")
        set_key(ENV_PATH, "GOOGLE_API_KEY", api_key)
        os.environ["GOOGLE_API_KEY"] = api_key
        print("API Key saved to .env and loaded into environment.")
        return True
    except Exception as e:
        print(f"Error saving API key: {e}")
        return False

def _record_loopback_thread(mic, samplerate, stop_event, chunks_list):
    """Records continuously from one loopback device until stop_event is set."""
    try:
        with mic.recorder(samplerate=samplerate) as recorder:
            while not stop_event.is_set():
                chunk = recorder.record(numframes=samplerate // 10)
                chunks_list.append(chunk)
    except Exception:
        pass  # Skip devices that are unavailable



def record_audio(filename="recording.wav", samplerate=48000):
    print("\n--- Audio Recording ---")

    loopbacks = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
    if not loopbacks:
        print("No loopback devices found. Cannot record system audio.")
        return False

    print(f"Recording from {len(loopbacks)} loopback device(s) simultaneously:")
    for m in loopbacks:
        print(f"  - {m.name}")

    stop_event = threading.Event()
    all_chunks = [[] for _ in loopbacks]
    threads = []

    for i, mic in enumerate(loopbacks):
        t = threading.Thread(
            target=_record_loopback_thread,
            args=(mic, samplerate, stop_event, all_chunks[i]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    input("Recording... Press ENTER to stop.\n")
    stop_event.set()
    for t in threads:
        t.join()

    valid = [(i, chunks) for i, chunks in enumerate(all_chunks) if chunks]
    if not valid:
        print("No audio recorded.")
        return False

    # Concatenate each loopback, downmix to mono, then mix all together
    recordings = []
    min_len = None
    for i, chunks in valid:
        arr = np.concatenate(chunks, axis=0)    # (frames, channels)
        mono = arr.mean(axis=1, keepdims=True)  # downmix to mono
        recordings.append(mono)
        if min_len is None or mono.shape[0] < min_len:
            min_len = mono.shape[0]

    mixed = np.zeros((min_len, 1), dtype=np.float32)
    for mono in recordings:
        mixed += mono[:min_len]

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    print(f"Saving to {filename}...")
    sf.write(filename, mixed, samplerate)
    print("Recording saved.")
    return True

def transcribe_audio(audio_file):
    api_key = get_api_key()
    if not api_key:
        print("API Key is missing. Please set it in Settings.")
        return None

    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        return None

    print("\n--- Transcribing ---")
    print("Configuring Gemini...")
    genai.configure(api_key=api_key)
    
    print(f"Uploading {audio_file}...")
    try:
        myfile = genai.upload_file(audio_file)
        print(f"Uploaded: {myfile.name}")
        
        print("Requesting transcription...")
        model = genai.GenerativeModel("gemini-2.0-flash")
        result = model.generate_content(
            [myfile, "Transcribe this audio file into text. Return the transcription in the same language as the audio. Do not translate."]
        )
        return result.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def main_menu():
    while True:
        print("\n=== Audio Transcriber CLI ===")
        print(f"API Key Status: {'SET' if os.environ.get('GOOGLE_API_KEY') else 'NOT SET'}")
        print("1. Record System Audio")
        print("2. Transcribe Audio File")
        print("3. View Transcript")
        print("4. Settings (Set API Key)")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            filename = input("Enter filename to save (default: system_audio.wav): ").strip() or "system_audio.wav"
            record_audio(filename)
            
        elif choice == '2':
            filename = input("Enter filename to transcribe (default: system_audio.wav): ").strip() or "system_audio.wav"
            transcript = transcribe_audio(filename)
            if transcript:
                print("\n--- Transcript ---")
                print(transcript)
                print("------------------")
                
                # Save option
                save_opt = input("Save transcript to file? (y/n): ").lower()
                if save_opt == 'y':
                    out_name = input("Output filename (default: transcript.txt): ").strip() or "transcript.txt"
                    try:
                        with open(out_name, "w", encoding="utf-8") as f:
                            f.write(transcript)
                        print(f"Saved to {out_name}")
                    except Exception as e:
                        print(f"Error saving file: {e}")

        elif choice == '3':
            filename = input("Enter filename to view (default: transcript.txt): ").strip() or "transcript.txt"
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        print(f"\n--- {filename} ---")
                        print(f.read())
                        print("------------------")
                except Exception as e:
                    print(f"Error reading file: {e}")
            else:
                print("File not found.")

        elif choice == '4':
            new_key = input("Enter Google API Key: ").strip()
            if new_key:
                save_api_key(new_key)
            else:
                print("No key entered.")

        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
