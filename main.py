import os
import sys
import threading
import soundcard as sc
import soundfile as sf
import google.generativeai as genai
import numpy as np
import warnings
import time
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

    # Get all mics and loopbacks
    all_mics = sc.all_microphones(include_loopback=True)
    
    # Filter logic:
    # 1. Include specific loopbacks (Surface Speakers, Jabra)
    # 2. Include specific real microphones (Surface Mic, Jabra Mic)
    # 3. Exclude Display Audio (U28E590)
    
    selected_devices = []
    print("Searching for appropriate audio sources...")
    
    for m in all_mics:
        name = m.name.lower()
        # Exclusion
        if "u28e590" in name:
            continue
            
        # Inclusion: Jabra, Surface, or the default system loopback
        # We look for "jabra", "surface", "speakers", "headphones", "headset"
        keywords = ["jabra", "surface", "speaker", "headphones", "headset", "microphone"]
        if any(kw in name for kw in keywords):
            selected_devices.append(m)
            type_str = "Loopback" if m.isloopback else "Mic"
            print(f"  - Found: {m.name} ({type_str})")

    if not selected_devices:
        print("No specific devices found. Using all available devices.")
        selected_devices = [m for m in all_mics if not "u28e590" in m.name.lower()]

    if not selected_devices:
        print("No audio devices found at all.")
        return False

    stop_event = threading.Event()
    all_chunks = [[] for _ in selected_devices]
    threads = []

    for i, mic in enumerate(selected_devices):
        t = threading.Thread(
            target=_record_loopback_thread,
            args=(mic, samplerate, stop_event, all_chunks[i]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    print("\nRecording... Press ENTER to stop.")
    print("(Pegel-Überwachung aktiv - Drücke ENTER zum Beenden)")
    
    time.sleep(0.5) # Give threads a moment to start filling chunks
    
    try:
        while not stop_event.is_set():
            # Check if user pressed ENTER
            import msvcrt
            if msvcrt.kbhit():
                char = msvcrt.getch()
                if char in [b'\r', b'\n']:
                    stop_event.set()
                    break
            
            # Simple volume display from any device
            current_peak = 0
            dev_name = ""
            for i in range(len(selected_devices)):
                if len(all_chunks[i]) > 0:
                    last_chunk = all_chunks[i][-1]
                    peak = np.max(np.abs(last_chunk))
                    if peak > current_peak:
                        current_peak = peak
                        dev_name = selected_devices[i].name
            
            # Draw a small bar
            bars = int(min(current_peak, 1.0) * 20)
            print(f"\rPegel: [{'#' * bars}{' ' * (20 - bars)}] {current_peak:.4f} ({dev_name[:20]})", end="", flush=True)
            time.sleep(0.1)
    except Exception as e:
        print(f"\nMonitor error: {e}")
        stop_event.set()

    print("\nStopping recording...")
    for t in threads:
        t.join()

    valid = [(i, chunks) for i, chunks in enumerate(all_chunks) if chunks]
    if not valid:
        print("No audio recorded.")
        return False

    # Concatenate each source, downmix to mono, then mix all together
    recordings = []
    max_len = 0
    print("\nProcessing audio streams:")
    for i, chunks in valid:
        arr = np.concatenate(chunks, axis=0)    # (frames, channels)
        mono = arr.mean(axis=1, keepdims=True)  # downmix to mono
        
        # Diagnostic: Check the volume levels of this device
        peak_vol = np.max(np.abs(mono))
        print(f"  - {selected_devices[i].name}: Peak Volume = {peak_vol:.4f}")
        
        recordings.append(mono)
        if mono.shape[0] > max_len:
            max_len = mono.shape[0]

    if max_len == 0:
        print("All recorded streams are empty.")
        return False

    mixed = np.zeros((max_len, 1), dtype=np.float32)
    for mono in recordings:
        length = mono.shape[0]
        mixed[:length] += mono

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    print(f"Final mixed peek volume: {peak:.4f}")
    if peak > 1.0:
        mixed /= peak
    elif peak < 0.0001:
        print("Warning: Final recording seems to be silent!")

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
        
        # Try to find a valid model dynamically to avoid 404
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model = "models/gemini-1.5-flash"
        if target_model not in available_models:
            flash_models = [m for m in available_models if 'flash' in m.lower()]
            if flash_models:
                target_model = flash_models[0]
            elif available_models:
                target_model = available_models[0]
        
        print(f"Using model: {target_model}")
        model = genai.GenerativeModel(target_model) 
        
        prompt = (
            "Transkribiere diese Audiodatei wortwörtlich in Text. "
            "Es handelt sich um ein Meeting oder Gespräch. "
            "Gib das Transkript in der Originalsprache des Audios zurück (Deutsch). "
            "Achte auf Vollständigkeit und ordne, wenn möglich, Sprecherwechsel grob ein."
        )
        result = model.generate_content([myfile, prompt])
        return result.text
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "leaked" in error_msg.lower():
            print("\n[!] FEHLER: Dein API-Key wurde als 'leaked' gemeldet und gesperrt.")
            print("Bitte erstelle einen neuen Key: https://aistudio.google.com/app/apikey")
            print("Aktualisiere den Key danach in den Einstellungen (Punkt 5).")
        else:
            print(f"Error during transcription: {e}")
        return None

def generate_protocol(transcript):
    api_key = get_api_key()
    if not api_key:
        print("API Key is missing.")
        return None

    if not transcript:
        print("No transcript provided.")
        return None

    print("\n--- Generating Protocol ---")
    genai.configure(api_key=api_key)
    
    # Try to find a valid model dynamically to avoid 404
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    target_model = "models/gemini-1.5-flash"
    if target_model not in available_models:
        flash_models = [m for m in available_models if 'flash' in m.lower()]
        if flash_models:
            target_model = flash_models[0]
        elif available_models:
            target_model = available_models[0]
    
    print(f"Using model: {target_model}")
    model = genai.GenerativeModel(target_model)
    
    prompt = (
        "Erstelle aus dem folgenden Transkript ein professionelles Sitzungsprotokoll in deutscher Sprache.\n\n"
        "Anforderungen:\n"
        "1. **Betreff**: Erstelle eine aussagekräftige Überschrift für das Treffen.\n"
        "2. **### Zusammenfassung**: Fasse die wichtigsten Kernpunkte in 3-5 prägnanten Sätzen zusammen. Vermeide Füllwörter.\n"
        "3. **### Aufgaben & Action Items**: Erstelle eine markdown-Tabelle mit den Spalten: | Aufgabe | Wer | Bis wann |. "
        "Extrahiere die Aufgaben so präzise wie möglich aus dem Kontext.\n\n"
        "WICHTIG: Verwende keine endlosen Trennstriche (wie '---') als Platzhalter. Wenn Informationen fehlen, lass die Felder leer oder schreib 'Nicht erwähnt'.\n\n"
        "Transkript:\n"
        f"{transcript}"
    )

    try:
        result = model.generate_content(prompt)
        return result.text
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "leaked" in error_msg.lower():
            print("\n[!] FEHLER: Dein API-Key ist gesperrt (leaked).")
            print("Bitte erneuere ihn unter: https://aistudio.google.com/app/apikey")
        else:
            print(f"Error during protocol generation: {e}")
        return None

def main_menu():
    current_transcript = ""
    current_protocol = ""
    
    while True:
        print("\n=== Audio Transcriber CLI ===")
        print(f"API Key Status: {'SET' if os.environ.get('GOOGLE_API_KEY') else 'NOT SET'}")
        print("1. Record System Audio")
        print("2. Transcribe Audio File")
        print("3. View Transcript / Generate Protocol")
        print("4. Save (Transcript/Protocol)")
        print("5. Settings (Set API Key)")
        print("6. Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == '1':
            filename = input("Enter filename to save (default: system_audio.wav): ").strip() or "system_audio.wav"
            record_audio(filename)
            
        elif choice == '2':
            filename = input("Enter filename to transcribe (default: system_audio.wav): ").strip() or "system_audio.wav"
            transcript = transcribe_audio(filename)
            if transcript:
                current_transcript = transcript
                current_protocol = "" # Reset protocol for new transcript
                print("\n--- Transcript ---")
                print(transcript)
                print("------------------")
                
                gen_prot = input("\nProtokoll aus diesem Transkript generieren? (y/n): ").lower()
                if gen_prot == 'y':
                    protocol = generate_protocol(transcript)
                    if protocol:
                        current_protocol = protocol # FIXED: persist for Option 5
                        print("\n--- Protocol ---")
                        print(protocol)
                        print("----------------")

        elif choice == '3':
            filename = input("Enter filename to view (default: transcript.txt): ").strip() or "transcript.txt"
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        content = f.read()
                        print(f"\n--- {filename} ---")
                        print(content)
                        print("------------------")
                        
                        # Store as current for Option 4 (Save)
                        current_transcript = content
                        
                        gen_prot = input("\nProtokoll aus dieser Datei generieren? (y/n): ").lower()
                        if gen_prot == 'y':
                            protocol = generate_protocol(content)
                            if protocol:
                                current_protocol = protocol
                                print("\n--- Protocol ---")
                                print(protocol)
                                print("----------------")
                except Exception as e:
                    print(f"Error reading file: {e}")
            else:
                print("File not found.")

        elif choice == '4':
            if not current_transcript and not current_protocol:
                print("Nothing to save. Please transcribe something first.")
                continue
                
            out_name = input("Output filename (default: protocol.md): ").strip() or "protocol.md"
            try:
                with open(out_name, "w", encoding="utf-8") as f:
                    if current_protocol:
                        f.write("# Meeting Protocol\n\n")
                        f.write(current_protocol)
                        if current_transcript:
                            f.write("\n\n--- Original Transcript ---\n\n")
                            f.write(current_transcript)
                    else:
                        f.write(current_transcript)
                print(f"Successfully saved to {out_name}")
            except Exception as e:
                print(f"Error saving file: {e}")

        elif choice == '5':
            new_key = input("Enter Google API Key: ").strip()
            if new_key:
                save_api_key(new_key)
            else:
                print("No key entered.")

        elif choice == '6':
            print("Exiting...")
            break
        
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
