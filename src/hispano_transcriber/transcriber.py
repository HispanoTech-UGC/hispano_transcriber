"""
Transcriptor simple de voz en español usando Vosk.
"""

import sounddevice as sd
import queue
import json
import sys
import os
from vosk import Model, KaldiRecognizer

# --- CONFIGURACIÓN ---
MODEL_PATH = "vosk-model-es-0.42/vosk-model-es-0.42"  # Ruta corregida del modelo
SAMPLE_RATE = 16000
BLOCK_SIZE = 3200  # 0.2 segundos por bloque

# --- COLA DE AUDIO ---
audio_queue = queue.Queue()


def callback(indata, frames, time, status):
    """Callback para captura de audio"""
    if status:
        print("⚠️", status, file=sys.stderr)
    audio_queue.put(bytes(indata))


def main():
    """Función principal del transcriptor"""
    # --- CARGA DEL MODELO ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modelo no encontrado en: {MODEL_PATH}")
        sys.exit(1)

    print("📦 Cargando modelo Vosk...")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    print("✅ Modelo cargado.\n")

    print("🎙️ Escuchando... pulsa Ctrl+C para salir.\n")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype='int16',
        channels=1,
        callback=callback
    ):
        try:
            while True:
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"\n🗣️ {text}\n")
                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text:
                        print(f"\r⌛ {partial_text}...", end="", flush=True)

        except KeyboardInterrupt:
            print("\n🛑 Finalizado.")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
