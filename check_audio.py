import soundcard as sc

print("=== Alle Lautsprecher (Speakers) ===")
for sp in sc.all_speakers():
    print(f"  Speaker: {sp.name}")

print()
print("=== Standard Lautsprecher (Default Speaker) ===")
default_sp = sc.default_speaker()
print(f"  {default_sp.name}")

print()
print("=== Alle Mikrofone inkl. Loopback ===")
for m in sc.all_microphones(include_loopback=True):
    tag = "LOOPBACK" if m.isloopback else "MIC    "
    print(f"  [{tag}] {m.name}")

print()
print("=== Standard Mikrofon ===")
default_mic = sc.default_microphone()
print(f"  {default_mic.name}")

print()
print("=== Analyse: Was wird aufgenommen? ===")
default_sp_name = default_sp.name
found_loopback = None
for m in sc.all_microphones(include_loopback=True):
    if m.name == default_sp_name:
        found_loopback = m
        break

if found_loopback:
    print(f"  OK - Loopback-Geraet gefunden: '{found_loopback.name}'")
    print(f"  => Systemton des Standard-Lautsprechers wird aufgenommen.")
else:
    print(f"  WARNUNG - Kein Loopback fuer '{default_sp_name}' gefunden!")
    print(f"  => Fallback auf Standard-Mikrofon: '{default_mic.name}'")
    print(f"  => Es wird nur das Mikrofon aufgenommen, KEIN Systemton!")
