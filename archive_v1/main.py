import os
import sys
from dotenv import load_dotenv

load_dotenv()
import threading
import soundcard as sc
import soundfile as sf
import google.generativeai as genai
import numpy as np

def get_api_key():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in environment variables.")
        api_key = input("Please enter your Google Gemini API Key: ").strip()
    return api_key

def record_audio(filename="recording.wav", samplerate=48000):
    print("Looking for loopback device (System Audio)...")
    
    # Try to find the default speaker's loopback
    # On Windows, soundcard usually exposes loopback devices via include_loopback=True
    
    default_speaker = sc.default_speaker()
    print(f"Default Speaker: {default_speaker.name}")
    
    # mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
    # The above might fail if exact name match isn't found or id isn't name. 
    # Let's search in all mics.
    
    loopback_mic = None
    try:
        mics = sc.all_microphones(include_loopback=True)
        for m in mics:
            # On Windows, loopback devices often have the same name as the speaker
            # or contain "Stereo Mix" or similar.
            # But soundcard usually maps the loopback of the speaker to a mic with the same name/id logic if include_loopback=True
            if m.name == default_speaker.name:
                loopback_mic = m
                break
        
        if not loopback_mic:
            # Fallback: try to find the first one that is loopback-like? 
            # soundcard doesn't explicitly flag loopback availability in metadata always.
            # Let's just try the default microphone if loopback not found, but warn.
            print("Warning: Could not strictly identify default speaker loopback. Using default microphone (might be your voice, not system audio).")
            # attempt to use the one with same matching ID if possible
            loopback_mic = sc.default_microphone()
    except Exception as e:
        print(f"Error finding mics: {e}")
        return

    print(f"Recording from: {loopback_mic.name}")
    
    data = []
    is_recording = True

    def record_loop():
        with loopback_mic.recorder(samplerate=samplerate) as mic:
            while is_recording:
                # Record in chunks
                chunk = mic.record(numframes=samplerate//10) # 0.1s chunks
                data.append(chunk)

    msg_thread = threading.Thread(target=record_loop)
    msg_thread.start()

    print("Recording... Press ENTER to stop.")
    input()
    is_recording = False
    msg_thread.join()

    # Concatenate all chunks
    if not data:
        print("No audio recorded.")
        return False
        
    full_recording = np.concatenate(data, axis=0)
    
    print(f"Saving to {filename}...")
    sf.write(filename, full_recording, samplerate)
    return True

def transcribe_audio(audio_file, api_key):
    print("Configuring Gemini...")
    genai.configure(api_key=api_key)
    
    print(f"Uploading {audio_file} to Gemini...")
    myfile = genai.upload_file(audio_file)
    print(f"Uploaded: {myfile.name}")
    
    print("Requesting transcription...")
    
    model_name = "gemini-2.0-flash"
    try:
        model = genai.GenerativeModel(model_name)
        result = model.generate_content(
            [myfile, "Transcribe this audio file into text. Return the transcription in the same language as the audio (e.g. German). Do not translate."]
        )
        return result.text
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        print("\nListing available models...")
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(m.name)
        except Exception as list_e:
            print(f"Could not list models: {list_e}")
        raise e

def main():
    api_key = get_api_key()
    
    audio_filename = "system_audio.wav"
    
    if record_audio(audio_filename):
        print("Recording finished.")
        try:
            transcript = transcribe_audio(audio_filename, api_key)
            output_file = "transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Transcription saved to {output_file}")
            print("-" * 20)
            print(transcript)
            print("-" * 20)
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
    else:
        print("Recording failed or was empty.")

if __name__ == "__main__":
    main()
