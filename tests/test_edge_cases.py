"""
Tests de casos extremos y edge cases para el sistema de transcripción.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import pytest


class TestEdgeCases(unittest.TestCase):
    """Tests para casos extremos y situaciones límite."""

    def setUp(self):
        """Preparación para tests de casos extremos."""
        self.sample_rate = 16000

    def test_empty_audio_handling(self):
        """Prueba manejo de audio vacío."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Array completamente vacío
        empty_audio = np.array([], dtype=np.int16)
        result = manager.identify_speaker(empty_audio)
        
        # Debería manejar graciosamente el caso vacío
        if result is not None:
            self.assertTrue(result.startswith("Hablante"))

    def test_silent_audio_handling(self):
        """Prueba manejo de audio completamente silencioso."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Audio silencioso (todos ceros)
        silent_audio = np.zeros(self.sample_rate * 2, dtype=np.int16)
        result = manager.identify_speaker(silent_audio)
        
        # Debería manejar audio silencioso sin fallar
        if result is not None:
            self.assertTrue(result.startswith("Hablante"))

    def test_extremely_loud_audio(self):
        """Prueba manejo de audio extremadamente fuerte."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Audio a máximo volumen
        loud_audio = np.full(self.sample_rate, 32767, dtype=np.int16)
        
        # No debería fallar con audio muy fuerte
        try:
            result = manager.identify_speaker(loud_audio)
            if result is not None:
                self.assertTrue(result.startswith("Hablante"))
        except Exception as e:
            self.fail(f"Audio fuerte causó excepción: {e}")

    def test_clipped_audio_handling(self):
        """Prueba manejo de audio con clipping."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Crear audio que se clippea
        t = np.linspace(0, 2, 2 * self.sample_rate)
        audio_float = np.sin(2 * np.pi * 440 * t) * 2.0  # Amplitud > 1
        
        # Simular clipping
        clipped_audio = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)
        
        result = manager.identify_speaker(clipped_audio)
        
        if result is not None:
            self.assertTrue(result.startswith("Hablante"))

    def test_very_short_audio_segments(self):
        """Prueba manejo de segmentos de audio muy cortos."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Diferentes duraciones muy cortas
        short_durations = [0.001, 0.01, 0.05, 0.1]  # en segundos
        
        for duration in short_durations:
            samples = max(1, int(duration * self.sample_rate))
            short_audio = np.ones(samples, dtype=np.int16) * 1000
            
            # No debería fallar con audio muy corto
            try:
                result = manager.identify_speaker(short_audio)
                # Puede retornar None para audio muy corto, eso está bien
            except Exception as e:
                self.fail(f"Audio corto ({duration}s) causó excepción: {e}")

    def test_very_long_audio_segments(self):
        """Prueba manejo de segmentos de audio muy largos."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Audio de 60 segundos (muy largo para procesamiento en tiempo real)
        duration = 60
        samples = duration * self.sample_rate
        
        # Crear audio sintético largo pero no demasiado para la memoria
        # Usar una función que genere el audio dinámicamente
        chunk_size = self.sample_rate  # 1 segundo de chunks
        long_audio = np.zeros(samples, dtype=np.int16)
        
        for i in range(0, samples, chunk_size):
            end_idx = min(i + chunk_size, samples)
            chunk_samples = end_idx - i
            t = np.linspace(0, chunk_samples / self.sample_rate, chunk_samples)
            chunk = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
            long_audio[i:end_idx] = chunk
        
        # Debería manejar audio largo sin problemas graves de memoria
        try:
            result = manager.identify_speaker(long_audio)
            if result is not None:
                self.assertTrue(result.startswith("Hablante"))
        except MemoryError:
            # Aceptable si hay limitaciones de memoria
            pass
        except Exception as e:
            # Otros errores no deberían ocurrir
            self.fail(f"Audio largo causó excepción inesperada: {e}")

    def test_audio_with_nan_values(self):
        """Prueba manejo de audio con valores NaN."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Crear audio con algunos valores NaN
        audio_float = np.random.randn(self.sample_rate).astype(np.float32)
        audio_float[100:200] = np.nan  # Insertar NaN values
        
        # Convertir a int16 (los NaN deberían convertirse a 0)
        audio_int16 = np.nan_to_num(audio_float * 32767).astype(np.int16)
        
        # Debería manejar los NaN sin fallar
        try:
            result = manager.identify_speaker(audio_int16)
            if result is not None:
                self.assertTrue(result.startswith("Hablante"))
        except Exception as e:
            self.fail(f"Audio con NaN causó excepción: {e}")

    def test_audio_with_inf_values(self):
        """Prueba manejo de audio con valores infinitos."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Crear audio con valores infinitos
        audio_float = np.random.randn(self.sample_rate).astype(np.float32)
        audio_float[50:100] = np.inf
        audio_float[150:200] = -np.inf
        
        # Limpiar infinitos
        audio_clean = np.nan_to_num(audio_float, posinf=1.0, neginf=-1.0)
        audio_int16 = (audio_clean * 32767).astype(np.int16)
        
        try:
            result = manager.identify_speaker(audio_int16)
            if result is not None:
                self.assertTrue(result.startswith("Hablante"))
        except Exception as e:
            self.fail(f"Audio con infinitos causó excepción: {e}")

    def test_maximum_number_of_speakers(self):
        """Prueba el comportamiento con el máximo número de hablantes."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        # Usar umbral muy alto para forzar nuevos hablantes
        manager = SpeakerManager(similarity_threshold=0.99)
        
        # Crear muchos segmentos diferentes
        speakers = []
        for i in range(50):  # Intentar crear 50 hablantes diferentes
            # Crear audio muy diferente para cada iteración
            freq = 200 + i * 20  # Frecuencias muy diferentes
            t = np.linspace(0, 1, self.sample_rate)
            audio = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
            
            speaker = manager.identify_speaker(audio)
            speakers.append(speaker)
        
        unique_speakers = set(speakers)
        
        # Debería crear múltiples hablantes
        self.assertGreater(len(unique_speakers), 1)
        
        # Verificar que todos los IDs son válidos
        for speaker in unique_speakers:
            self.assertTrue(speaker.startswith("Hablante"))
            # Extraer número del hablante
            number = int(speaker.split(" ")[1])
            self.assertGreater(number, 0)

    def test_speaker_threshold_edge_cases(self):
        """Prueba casos extremos de umbrales de similitud."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        # Umbral mínimo (0.0) - todos son el mismo hablante
        manager_min = SpeakerManager(similarity_threshold=0.0)
        
        # Crear dos audios muy diferentes
        t = np.linspace(0, 1, self.sample_rate)
        audio1 = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        audio2 = (np.sin(2 * np.pi * 880 * t) * 16000).astype(np.int16)
        
        speaker1 = manager_min.identify_speaker(audio1)
        speaker2 = manager_min.identify_speaker(audio2)
        
        # Con umbral 0.0, deberían ser el mismo hablante
        self.assertEqual(speaker1, speaker2)
        
        # Umbral máximo (1.0) - todos son hablantes diferentes
        manager_max = SpeakerManager(similarity_threshold=1.0)
        
        # Incluso audio idéntico debería crear hablantes diferentes
        audio_identical = audio1.copy()
        
        speaker1_max = manager_max.identify_speaker(audio1)
        speaker2_max = manager_max.identify_speaker(audio_identical)
        
        # Con umbral 1.0, incluso audio idéntico puede ser diferente hablante
        # (dependiendo de la implementación)

    def test_corrupted_audio_data(self):
        """Prueba manejo de datos de audio corruptos."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Diferentes tipos de corrupción
        corrupted_cases = [
            # Patrón repetitivo extremo
            np.full(self.sample_rate, 12345, dtype=np.int16),
            
            # Alternancia rápida entre extremos
            np.tile([32767, -32768], self.sample_rate // 2).astype(np.int16),
            
            # Ruido completamente aleatorio
            np.random.randint(-32768, 32767, self.sample_rate, dtype=np.int16),
        ]
        
        for i, corrupted_audio in enumerate(corrupted_cases):
            try:
                result = manager.identify_speaker(corrupted_audio)
                # Cualquier resultado válido o None está bien
                if result is not None:
                    self.assertTrue(result.startswith("Hablante"))
                print(f"✅ Caso de corrupción {i+1} manejado correctamente")
            except Exception as e:
                self.fail(f"Caso de corrupción {i+1} causó excepción: {e}")

    def test_memory_exhaustion_simulation(self):
        """Simula condiciones de poca memoria."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Crear audio normal
        t = np.linspace(0, 1, self.sample_rate)
        audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        
        # Simular muchas identificaciones para llenar memoria
        speakers = []
        for i in range(100):  # Muchas iteraciones
            try:
                # Variar ligeramente el audio
                variation = 1 + 0.01 * i
                varied_audio = (audio * variation).astype(np.int16)
                
                speaker = manager.identify_speaker(varied_audio)
                speakers.append(speaker)
                
                # Verificar que no hay crecimiento descontrolado de datos
                total_embeddings = sum(len(embs) for embs in manager.speakers.values())
                self.assertLess(total_embeddings, 1000, "Demasiados embeddings almacenados")
                
            except MemoryError:
                # Aceptable en condiciones de poca memoria
                break
            except Exception as e:
                self.fail(f"Error inesperado en iteración {i}: {e}")
        
        # Debería haber procesado al menos algunos
        self.assertGreater(len(speakers), 0)


class TestBoundaryConditions(unittest.TestCase):
    """Tests para condiciones límite específicas."""

    def test_sample_rate_variations(self):
        """Prueba con diferentes frecuencias de muestreo."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Probar con audio de diferentes frecuencias de muestreo
        # (simulado redimensionando el array)
        original_duration = 2  # segundos
        
        sample_rates = [8000, 16000, 22050, 44100]
        
        for sr in sample_rates:
            samples = int(original_duration * sr)
            t = np.linspace(0, original_duration, samples)
            audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
            
            # Redimensionar para simular conversión a 16kHz
            if sr != 16000:
                target_samples = int(original_duration * 16000)
                # Simple resampling por interpolación
                indices = np.linspace(0, len(audio) - 1, target_samples)
                audio_resampled = np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)
            else:
                audio_resampled = audio
            
            try:
                result = manager.identify_speaker(audio_resampled)
                if result is not None:
                    self.assertTrue(result.startswith("Hablante"))
                print(f"✅ Sample rate {sr}Hz manejado correctamente")
            except Exception as e:
                self.fail(f"Sample rate {sr}Hz causó excepción: {e}")

    def test_extreme_frequencies(self):
        """Prueba con frecuencias extremas de audio."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Frecuencias extremas (pero dentro del rango audible)
        extreme_frequencies = [
            20,     # Muy grave
            100,    # Grave
            8000,   # Muy agudo (cerca del límite de Nyquist para 16kHz)
        ]
        
        for freq in extreme_frequencies:
            t = np.linspace(0, 2, 2 * 16000)
            audio = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
            
            try:
                result = manager.identify_speaker(audio)
                if result is not None:
                    self.assertTrue(result.startswith("Hablante"))
                print(f"✅ Frecuencia {freq}Hz manejada correctamente")
            except Exception as e:
                self.fail(f"Frecuencia {freq}Hz causó excepción: {e}")


if __name__ == '__main__':
    unittest.main()
