"""
Transcriptor de voz en español con identificación de hablantes.
"""

import sounddevice as sd
import queue
import json
import sys
import os
import numpy as np
from vosk import Model, KaldiRecognizer
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
from sklearn.metrics.pairwise import cosine_similarity
import time
import argparse

# --- CONFIGURACIÓN ---
MODEL_PATH = "vosk-model-es-0.42"
SAMPLE_RATE = 16000
BLOCK_SIZE = 3200  # 0.2 segundos por bloque
SPEAKER_BUFFER_DURATION = 3.0  # Segundos de audio para análisis de hablante
MIN_SPEECH_DURATION = 1.0  # Duración mínima para considerar como habla

# --- VARIABLES GLOBALES ---
audio_queue = queue.Queue()
speaker_embeddings = []
speaker_labels = []
current_speaker = "Hablante 1"
speaker_count = 0
audio_buffer = []
last_speech_time = None

# --- CONFIGURACIÓN DE IDENTIFICACIÓN DE HABLANTE ---
USE_SIMPLE_FEATURES = False  # Usar características simples en lugar de Wav2Vec2
processor = None
wav2vec_model = None


class SpeakerManager:
    def __init__(self, similarity_threshold=0.85):
        self.speakers = {}  # {speaker_id: [embeddings]}
        self.speaker_count = 0
        self.similarity_threshold = similarity_threshold
        
    def extract_voice_features(self, audio_segment):
        """Extrae características simples de voz"""
        try:
            # Convertir a float32 y normalizar
            audio_float = audio_segment.astype(np.float32) / 32768.0
            
            if len(audio_float) < int(0.1 * SAMPLE_RATE):  # Muy corto
                return None
            
            # 1. Energía RMS
            rms = np.sqrt(np.mean(np.square(audio_float)))
            
            # 2. Tasa de cruces por cero
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_float)))) / 2
            zcr = zero_crossings / len(audio_float)
            
            # 3. Algunas características espectrales simples
            if len(audio_float) >= 512:
                spectrum = np.abs(np.fft.rfft(audio_float[:512]))
                spectral_centroid = np.sum(spectrum * np.arange(len(spectrum))) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0
                spectral_rolloff = np.searchsorted(np.cumsum(spectrum) / np.sum(spectrum), 0.85) if np.sum(spectrum) > 0 else 0
            else:
                spectral_centroid = 0
                spectral_rolloff = 0
            
            # Crear vector de características
            features = np.array([rms, zcr, spectral_centroid, spectral_rolloff])
            
            # Manejar NaN values
            features = np.nan_to_num(features, nan=0.0)
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error extrayendo características: {e}")
            return None
    
    def extract_embedding(self, audio_segment):
        """Extrae embedding de un segmento de audio"""
        # Si estamos usando características simples
        if USE_SIMPLE_FEATURES or processor is None or wav2vec_model is None:
            return self.extract_voice_features(audio_segment)
            
        try:
            # Normalizar audio
            audio_segment = audio_segment.astype(np.float32)
            if len(audio_segment) < SAMPLE_RATE:  # Muy corto
                return None
                
            # Procesar con Wav2Vec2
            inputs = processor(audio_segment, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = wav2vec_model(**inputs)
                # Usar la media del hidden state como embedding
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
            return embedding
        except Exception as e:
            print(f"⚠️ Error extrayendo embedding: {e}")
            # Intentar con características simples como fallback
            return self.extract_voice_features(audio_segment)
    
    def identify_speaker(self, audio_segment):
        """Identifica el hablante de un segmento de audio"""
        embedding = self.extract_embedding(audio_segment)
        if embedding is None:
            print("[DEBUG] No se pudo extraer embedding/características. Se asigna nuevo hablante.")
            return f"Hablante {self.speaker_count + 1}"
        
        if len(self.speakers) == 0:
            self.speaker_count += 1
            speaker_id = f"Hablante {self.speaker_count}"
            self.speakers[speaker_id] = [embedding]
            print(f"[DEBUG] Primer hablante detectado: {speaker_id}")
            return speaker_id
        
        best_similarity = 0
        best_speaker = None
        debug_similarities = {}
        for speaker_id, embeddings in self.speakers.items():
            similarities = [cosine_similarity([embedding], [emb])[0][0] for emb in embeddings]
            avg_similarity = np.mean(similarities)
            debug_similarities[speaker_id] = avg_similarity
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_speaker = speaker_id
        print(f"[DEBUG] Similitudes calculadas: {debug_similarities}")
        print(f"[DEBUG] Mejor hablante: {best_speaker} (similitud={best_similarity:.3f}), umbral={self.similarity_threshold}")
        if best_similarity > self.similarity_threshold:
            self.speakers[best_speaker].append(embedding)
            if len(self.speakers[best_speaker]) > 5:
                self.speakers[best_speaker] = self.speakers[best_speaker][-5:]
            print(f"[DEBUG] Hablante identificado como EXISTENTE: {best_speaker}")
            return best_speaker
        else:
            self.speaker_count += 1
            speaker_id = f"Hablante {self.speaker_count}"
            self.speakers[speaker_id] = [embedding]
            print(f"[DEBUG] Hablante identificado como NUEVO: {speaker_id}")
            return speaker_id


def callback(indata, frames, time, status):
    """Callback para captura de audio"""
    if status:
        print("⚠️", status, file=sys.stderr)
    
    # Convertir buffer crudo a array de NumPy
    audio_data = np.frombuffer(indata, dtype=np.int16)

    # Agregar a la cola para Vosk
    audio_queue.put(bytes(audio_data))

    # Agregar al buffer para análisis de hablante
    global audio_buffer, last_speech_time
    audio_buffer.extend(audio_data.tolist())

    # Mantener buffer de duración fija
    max_buffer_size = int(SPEAKER_BUFFER_DURATION * SAMPLE_RATE)
    if len(audio_buffer) > max_buffer_size:
        audio_buffer = audio_buffer[-max_buffer_size:]


def analyze_speaker_change(speaker_manager):
    """Analiza si ha cambiado el hablante"""
    global current_speaker, audio_buffer, last_speech_time
    
    if len(audio_buffer) < int(MIN_SPEECH_DURATION * SAMPLE_RATE):
        return current_speaker
    
    # Convertir buffer a numpy array
    audio_segment = np.array(audio_buffer, dtype=np.float32)
    
    # Identificar hablante
    identified_speaker = speaker_manager.identify_speaker(audio_segment)
    
    if identified_speaker != current_speaker:
        print(f"[DEBUG] Cambio de hablante: {current_speaker} -> {identified_speaker}")
        print(f"[DEBUG] Historial de hablantes:")
        for spk, feats in speaker_manager.speakers.items():
            print(f"  {spk}: {len(feats)} muestras")
        current_speaker = identified_speaker
        print(f"\n🔄 Cambio detectado → {current_speaker}")
    
    return current_speaker


def format_speaker_output(speaker, text):
    """Formatea la salida con información del hablante"""
    colors = {
        "Hablante 1": "🔵",
        "Hablante 2": "🔴", 
        "Hablante 3": "🟢",
        "Hablante 4": "🟡",
        "Hablante 5": "🟣"
    }
    
    icon = colors.get(speaker, "⚪")
    return f"{icon} {speaker}: {text}"


def cargar_modelos():
    """Carga los modelos necesarios para transcripción e identificación de hablantes"""
    global processor, wav2vec_model, USE_SIMPLE_FEATURES
    
    # --- CARGA DEL MODELO DE TRANSCRIPCIÓN ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modelo Vosk no encontrado en: {MODEL_PATH}")
        sys.exit(1)

    print("📦 Cargando modelo Vosk...")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    print("✅ Modelo Vosk cargado.")
    
    # --- CONFIGURACIÓN DE IDENTIFICACIÓN DE HABLANTE ---
    print("📦 Configurando identificación de hablantes...")
    
    # Solo intentamos cargar Wav2Vec2 si se necesita
    if not USE_SIMPLE_FEATURES:
        try:
            print("📦 Cargando modelo Wav2Vec2 para identificación de hablantes...")
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

            wav2vec_model.eval()
            print("✅ Modelo Wav2Vec2 cargado.")
        except Exception as e:
            print(f"⚠️ Error cargando Wav2Vec2: {e}")
            print("📦 Continuando con identificación simple de hablantes...")
            USE_SIMPLE_FEATURES = True
    
    return model, recognizer


def main():
    """Función principal del transcriptor con identificación de hablantes"""
    global current_speaker, last_speech_time
    
    # Configurar argparse
    parser = argparse.ArgumentParser(description='Transcriptor con identificación de hablantes')
    parser.add_argument('--umbral', type=float, default=0.85, 
                       help='Umbral de similitud para decidir si es el mismo hablante (0-1)')
    parser.add_argument('--simple', action='store_true', 
                       help='Usar características simples en vez de Wav2Vec2')
    args = parser.parse_args()
    
    # Configurar uso de características simples
    global USE_SIMPLE_FEATURES
    if args.simple:
        USE_SIMPLE_FEATURES = True
        print("🔧 Usando características simples para identificación de hablantes.")
    
    # Cargar modelos
    model, recognizer = cargar_modelos()
    
    # Inicializar gestor de hablantes
    speaker_manager = SpeakerManager(similarity_threshold=args.umbral)
    print(f"🔧 Umbral de similitud configurado en: {args.umbral}")
    
    # Inicializar tiempo
    last_speech_time = time.time()
    
    print("🎙️ Transcriptor con identificación de hablantes")
    print("🎯 Escuchando... pulsa Ctrl+C para salir.\n")
    
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
                    # Transcripción completa
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    
                    if text:
                        # Analizar cambio de hablante solo si hay texto
                        speaker = analyze_speaker_change(speaker_manager)
                        output = format_speaker_output(speaker, text)
                        print(f"\n{output}\n")
                        last_speech_time = time.time()
                        
                else:
                    # Transcripción parcial
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    
                    if partial_text:
                        # Mostrar transcripción parcial con hablante actual
                        print(f"\r⌛ {current_speaker}: {partial_text}...", end="", flush=True)

        except KeyboardInterrupt:
            print("\n🛑 Finalizado.")
            print(f"\n📊 Resumen de la sesión:")
            print(f"   Total de hablantes detectados: {speaker_manager.speaker_count}")
            for speaker_id in speaker_manager.speakers.keys():
                count = len(speaker_manager.speakers[speaker_id])
                print(f"   {speaker_id}: {count} segmentos de voz")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
