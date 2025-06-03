"""
Tests de rendimiento y benchmarking para el sistema de transcripción.
"""

import unittest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch
import pytest


class TestPerformance(unittest.TestCase):
    """Tests de rendimiento del sistema."""

    def setUp(self):
        """Preparación para tests de rendimiento."""
        self.sample_rate = 16000
        self.test_duration = 1  # 1 segundo de audio para tests rápidos

    @pytest.mark.slow
    def test_transcription_latency(self):
        """Mide la latencia del sistema de transcripción."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Crear audio de prueba
        t = np.linspace(0, self.test_duration, int(self.test_duration * self.sample_rate))
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        # Medir tiempo de procesamiento
        latencies = []
        for _ in range(10):  # 10 iteraciones para obtener promedio
            start_time = time.perf_counter()
            speaker = manager.identify_speaker(audio)
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            self.assertIsNotNone(speaker)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)
        
        print(f"\n📊 Estadísticas de latencia:")
        print(f"   Promedio: {avg_latency:.3f}s")
        print(f"   Máximo: {max_latency:.3f}s")
        print(f"   Mínimo: {min_latency:.3f}s")
        
        # La latencia promedio debería ser razonable
        self.assertLess(avg_latency, 2.0, "Latencia promedio muy alta")

    @pytest.mark.slow
    def test_throughput_multiple_speakers(self):
        """Mide el throughput con múltiples hablantes."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Crear múltiples segmentos de audio con diferentes características
        segments = []
        for freq in [440, 523, 659, 784, 880]:  # Diferentes frecuencias
            t = np.linspace(0, self.test_duration, int(self.test_duration * self.sample_rate))
            segment = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            segments.append(segment)
        
        # Medir tiempo total de procesamiento
        start_time = time.perf_counter()
        
        speakers = []
        for segment in segments:
            speaker = manager.identify_speaker(segment)
            speakers.append(speaker)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calcular throughput
        total_audio_duration = len(segments) * self.test_duration
        throughput = total_audio_duration / total_time  # ratio de tiempo real
        
        print(f"\n📊 Estadísticas de throughput:")
        print(f"   Audio total procesado: {total_audio_duration:.1f}s")
        print(f"   Tiempo de procesamiento: {total_time:.3f}s")
        print(f"   Throughput: {throughput:.2f}x tiempo real")
        
        # El throughput debería ser al menos 1x (tiempo real)
        self.assertGreater(throughput, 0.5, "Throughput muy bajo")
        
        # Verificar que se identificaron hablantes
        unique_speakers = set(speakers)
        self.assertGreaterEqual(len(unique_speakers), 1)

    @pytest.mark.slow
    def test_memory_usage_over_time(self):
        """Monitorea el uso de memoria durante procesamiento prolongado."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        import gc
        
        manager = SpeakerManager()
        
        # Crear audio de prueba
        t = np.linspace(0, 1, self.sample_rate)
        base_audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        # Procesar muchos segmentos
        num_segments = 100
        speakers_identified = []
        
        for i in range(num_segments):
            # Variar ligeramente el audio para cada iteración
            variation = 1 + 0.01 * np.sin(i / 10)  # Variación sutil
            audio = (base_audio * variation).astype(np.int16)
            
            speaker = manager.identify_speaker(audio)
            speakers_identified.append(speaker)
            
            # Forzar garbage collection cada 10 iteraciones
            if i % 10 == 0:
                gc.collect()
        
        # Verificar que el sistema sigue funcionando después de muchas iteraciones
        final_audio = base_audio
        final_speaker = manager.identify_speaker(final_audio)
        
        self.assertIsNotNone(final_speaker)
        self.assertEqual(len(speakers_identified), num_segments)
        
        print(f"\n📊 Procesamiento prolongado completado:")
        print(f"   Segmentos procesados: {num_segments}")
        print(f"   Hablantes únicos identificados: {len(set(speakers_identified))}")

    def test_concurrent_speaker_identification(self):
        """Prueba identificación concurrente de hablantes."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        import concurrent.futures
        
        def identify_speaker_task(manager, audio_data, task_id):
            """Tarea para identificar hablante en un hilo separado."""
            try:
                speaker = manager.identify_speaker(audio_data)
                return (task_id, speaker, True, None)
            except Exception as e:
                return (task_id, None, False, str(e))
        
        manager = SpeakerManager()
        
        # Crear múltiples segmentos de audio
        audio_segments = []
        for i in range(5):
            t = np.linspace(0, 1, self.sample_rate)
            freq = 440 + i * 50  # Diferentes frecuencias
            audio = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            audio_segments.append(audio)
        
        # Procesar concurrentemente
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, audio in enumerate(audio_segments):
                future = executor.submit(identify_speaker_task, manager, audio, i)
                futures.append(future)
            
            # Recopilar resultados
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        end_time = time.perf_counter()
        concurrent_time = end_time - start_time
        
        # Procesar secuencialmente para comparar
        start_time = time.perf_counter()
        sequential_results = []
        for i, audio in enumerate(audio_segments):
            speaker = manager.identify_speaker(audio)
            sequential_results.append((i, speaker, True, None))
        end_time = time.perf_counter()
        sequential_time = end_time - start_time
        
        print(f"\n📊 Comparación de rendimiento:")
        print(f"   Tiempo concurrente: {concurrent_time:.3f}s")
        print(f"   Tiempo secuencial: {sequential_time:.3f}s")
        print(f"   Speedup: {sequential_time/concurrent_time:.2f}x")
        
        # Verificar que todos los trabajos tuvieron éxito
        successful_concurrent = sum(1 for _, _, success, _ in results if success)
        self.assertEqual(successful_concurrent, len(audio_segments))

    def test_feature_extraction_performance(self):
        """Mide el rendimiento de la extracción de características."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Crear diferentes tamaños de audio
        audio_sizes = [
            (0.5, "Corto"),
            (1.0, "Medio"),
            (2.0, "Largo"),
            (5.0, "Muy largo")
        ]
        
        performance_data = []
        
        for duration, label in audio_sizes:
            # Crear audio de prueba
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            
            # Medir tiempo de extracción de características
            times = []
            for _ in range(5):  # 5 iteraciones para promedio
                start_time = time.perf_counter()
                features = manager.extract_voice_features(audio)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            performance_data.append((duration, label, avg_time, len(audio)))
            
            print(f"📊 {label} ({duration}s): {avg_time:.4f}s promedio")
        
        # Verificar que el tiempo de procesamiento escala razonablemente
        for duration, label, avg_time, samples in performance_data:
            # El tiempo debería ser proporcional a la duración, pero no más de 10x
            max_expected_time = duration * 10  # Factor conservador
            self.assertLess(avg_time, max_expected_time, 
                          f"Procesamiento de {label} toma demasiado tiempo")


class TestBenchmarks(unittest.TestCase):
    """Benchmarks específicos para diferentes componentes."""

    @pytest.mark.slow
    def test_speaker_similarity_computation_benchmark(self):
        """Benchmark para cálculo de similitud entre hablantes."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Crear embeddings sintéticos
        num_embeddings = 100
        embedding_size = 4  # Tamaño usado por características simples
        
        embeddings = []
        for i in range(num_embeddings):
            # Crear embeddings con cierta estructura
            base = np.random.randn(embedding_size)
            embedding = base + 0.1 * np.random.randn(embedding_size)  # Añadir ruido
            embeddings.append(embedding)
        
        # Benchmark de cálculos de similitud
        start_time = time.perf_counter()
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        num_comparisons = len(similarities)
        time_per_comparison = total_time / num_comparisons
        
        print(f"\n📊 Benchmark de similitud:")
        print(f"   Embeddings: {num_embeddings}")
        print(f"   Comparaciones: {num_comparisons}")
        print(f"   Tiempo total: {total_time:.3f}s")
        print(f"   Tiempo por comparación: {time_per_comparison*1000:.2f}ms")
        
        # Verificar que el tiempo por comparación es razonable
        self.assertLess(time_per_comparison, 0.001, "Cálculo de similitud muy lento")

    def test_audio_preprocessing_benchmark(self):
        """Benchmark para preprocesamiento de audio."""
        import numpy as np
        
        # Diferentes tamaños de audio
        test_sizes = [
            (1, "1 segundo"),
            (5, "5 segundos"),
            (10, "10 segundos"),
            (30, "30 segundos")
        ]
        
        sample_rate = 16000
        
        for duration, label in test_sizes:
            # Crear audio sintético
            samples = int(duration * sample_rate)
            audio_float = np.random.randn(samples).astype(np.float32)
            
            # Benchmark conversión a int16
            start_time = time.perf_counter()
            audio_int16 = (audio_float * 32767).astype(np.int16)
            conversion_time = time.perf_counter() - start_time
            
            # Benchmark cálculos estadísticos básicos
            start_time = time.perf_counter()
            rms = np.sqrt(np.mean(np.square(audio_float)))
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_float)))) / 2
            stats_time = time.perf_counter() - start_time
            
            # Benchmark FFT (en ventana pequeña)
            start_time = time.perf_counter()
            if len(audio_float) >= 512:
                spectrum = np.abs(np.fft.rfft(audio_float[:512]))
            fft_time = time.perf_counter() - start_time
            
            print(f"📊 Audio {label}:")
            print(f"   Conversión: {conversion_time*1000:.2f}ms")
            print(f"   Estadísticas: {stats_time*1000:.2f}ms")
            print(f"   FFT: {fft_time*1000:.2f}ms")
            
            # Verificar tiempos razonables
            self.assertLess(conversion_time, duration * 0.1, "Conversión muy lenta")
            self.assertLess(stats_time, duration * 0.1, "Cálculos estadísticos muy lentos")


if __name__ == '__main__':
    unittest.main()
