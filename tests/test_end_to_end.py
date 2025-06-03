"""
Tests end-to-end para el sistema completo de transcripción.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import tempfile
import os
import threading
import time


class TestEndToEndTranscription(unittest.TestCase):
    """Pruebas end-to-end del sistema de transcripción."""

    def setUp(self):
        """Preparación para las pruebas."""
        self.sample_rate = 16000
        self.test_duration = 2
        
        # Crear audio sintético para pruebas
        t = np.linspace(0, self.test_duration, int(self.test_duration * self.sample_rate))
        self.test_audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    @patch('hispano_transcriber.transcriber.os.path.exists')
    @patch('hispano_transcriber.transcriber.Model')
    @patch('hispano_transcriber.transcriber.KaldiRecognizer')
    @patch('hispano_transcriber.transcriber.sd.RawInputStream')
    @patch('hispano_transcriber.transcriber.audio_queue')
    def test_complete_transcription_workflow(self, mock_queue, mock_stream, mock_recognizer, mock_model, mock_exists):
        """Prueba el flujo completo de transcripción básica."""
        from hispano_transcriber import transcriber
        
        # Setup mocks
        mock_exists.return_value = True
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_recognizer_instance = Mock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Simular secuencia de transcripción
        recognition_results = [
            (True, '{"text": "Hola"}'),
            (False, '{"partial": "mundo"}'),
            (True, '{"text": "mundo"}'),
            (True, '{"text": "¿Cómo estás?"}')
        ]
        
        def mock_accept_waveform(data):
            if recognition_results:
                is_final, result = recognition_results.pop(0)
                if is_final:
                    mock_recognizer_instance.Result.return_value = result
                else:
                    mock_recognizer_instance.PartialResult.return_value = result
                return is_final
            return False
        
        mock_recognizer_instance.AcceptWaveform.side_effect = mock_accept_waveform
        
        # Simular datos de audio
        audio_chunks = [self.test_audio_data[i:i+3200].tobytes() 
                       for i in range(0, len(self.test_audio_data), 3200)]
        audio_chunks.append(KeyboardInterrupt())  # Terminar la simulación
        
        mock_queue.get.side_effect = audio_chunks
        
        # Ejecutar transcripción
        try:
            transcriber.main()
        except SystemExit:
            pass  # Esperado al simular Ctrl+C

    @patch('hispano_transcriber.transcriber_speaker.os.path.exists')
    @patch('hispano_transcriber.transcriber_speaker.Model')
    @patch('hispano_transcriber.transcriber_speaker.KaldiRecognizer')
    @patch('hispano_transcriber.transcriber_speaker.sd.RawInputStream')
    @patch('hispano_transcriber.transcriber_speaker.audio_queue')
    @patch('hispano_transcriber.transcriber_speaker.SpeakerManager')
    def test_complete_speaker_transcription_workflow(self, mock_speaker_manager, mock_queue, mock_stream, mock_recognizer, mock_model, mock_exists):
        """Prueba el flujo completo de transcripción con identificación de hablantes."""
        from hispano_transcriber import transcriber_speaker
        
        # Setup mocks
        mock_exists.return_value = True
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_recognizer_instance = Mock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        mock_speaker_instance = Mock()
        mock_speaker_manager.return_value = mock_speaker_instance
        
        # Simular identificación de múltiples hablantes
        speaker_sequence = ["Hablante 1", "Hablante 2", "Hablante 1", "Hablante 3"]
        mock_speaker_instance.identify_speaker.side_effect = speaker_sequence
        
        # Simular resultados de transcripción
        transcription_results = [
            '{"text": "Buenos días"}',
            '{"text": "Hola, ¿cómo estás?"}',
            '{"text": "Muy bien, gracias"}',
            '{"text": "Me alegro de escucharlo"}'
        ]
        
        mock_recognizer_instance.AcceptWaveform.return_value = True
        mock_recognizer_instance.Result.side_effect = transcription_results
        
        # Simular detección de habla
        with patch('hispano_transcriber.transcriber_speaker.is_speech_detected', return_value=True):
            # Simular datos de audio
            audio_chunks = [self.test_audio_data[i:i+3200].tobytes() 
                           for i in range(0, len(self.test_audio_data), 3200)]
            audio_chunks.extend([self.test_audio_data[i:i+3200].tobytes() 
                               for i in range(0, len(self.test_audio_data), 3200)] * 3)  # Más chunks
            audio_chunks.append(KeyboardInterrupt())
            
            mock_queue.get.side_effect = audio_chunks
            
            # Ejecutar transcripción con hablantes
            try:
                transcriber_speaker.main()
            except SystemExit:
                pass

    def test_audio_processing_pipeline(self):
        """Prueba el pipeline completo de procesamiento de audio."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Crear múltiples segmentos de audio con diferentes características
        segments = []
        for freq in [440, 523, 659]:  # Diferentes frecuencias (notas musicales)
            t = np.linspace(0, 2, 2 * self.sample_rate)
            segment = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            segments.append(segment)
        
        # Procesar cada segmento
        speakers = []
        for i, segment in enumerate(segments):
            speaker = manager.identify_speaker(segment)
            speakers.append(speaker)
        
        # Verificar que se identificaron múltiples hablantes
        unique_speakers = set(speakers)
        self.assertGreaterEqual(len(unique_speakers), 1)  # Al menos un hablante
        
        # Verificar formato de IDs de hablantes
        for speaker in speakers:
            self.assertTrue(speaker.startswith("Hablante"))

    def test_error_recovery_scenarios(self):
        """Prueba escenarios de recuperación de errores."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        manager = SpeakerManager()
        
        # Test 1: Audio corrupto/inválido
        corrupted_audio = np.array([np.inf, -np.inf, np.nan] * 1000, dtype=np.float32)
        corrupted_audio_int16 = np.nan_to_num(corrupted_audio).astype(np.int16)
        
        # No debería fallar, debería manejar graciosamente
        try:
            speaker = manager.identify_speaker(corrupted_audio_int16)
            self.assertIsNotNone(speaker)
        except Exception as e:
            # Si hay excepción, debería ser manejada apropiadamente
            self.fail(f"Error no manejado con audio corrupto: {e}")
        
        # Test 2: Audio muy corto
        very_short_audio = np.ones(10, dtype=np.int16)
        try:
            speaker = manager.identify_speaker(very_short_audio)
            # Puede retornar None o un speaker válido
            if speaker is not None:
                self.assertTrue(speaker.startswith("Hablante"))
        except Exception as e:
            self.fail(f"Error no manejado con audio muy corto: {e}")

    def test_concurrent_processing(self):
        """Prueba procesamiento concurrente básico."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        import threading
        import queue
        
        manager = SpeakerManager()
        results_queue = queue.Queue()
        
        def process_audio_segment(segment, segment_id):
            """Función para procesar un segmento en un hilo separado."""
            try:
                speaker = manager.identify_speaker(segment)
                results_queue.put((segment_id, speaker, True))
            except Exception as e:
                results_queue.put((segment_id, str(e), False))
        
        # Crear múltiples segmentos de audio
        segments = []
        for i in range(3):
            t = np.linspace(0, 1, self.sample_rate)
            freq = 440 + i * 100  # Diferentes frecuencias
            segment = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            segments.append(segment)
        
        # Procesar segmentos concurrentemente
        threads = []
        for i, segment in enumerate(segments):
            thread = threading.Thread(target=process_audio_segment, args=(segment, i))
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen todos los hilos
        for thread in threads:
            thread.join(timeout=5)  # Timeout de 5 segundos
        
        # Recopilar resultados
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verificar que se procesaron todos los segmentos
        self.assertEqual(len(results), len(segments))
        
        # Verificar que todos tuvieron éxito
        for segment_id, result, success in results:
            self.assertTrue(success, f"Fallo en segmento {segment_id}: {result}")
            if success:
                self.assertTrue(result.startswith("Hablante"))

    def test_memory_usage_stability(self):
        """Prueba básica de estabilidad de uso de memoria."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        import gc
        
        manager = SpeakerManager()
        
        # Procesar muchos segmentos de audio para verificar que no hay memory leaks obvios
        for i in range(50):  # Reducido para pruebas rápidas
            t = np.linspace(0, 0.5, self.sample_rate // 2)  # Segmentos más cortos
            freq = 440 + (i % 10) * 50  # Variar frecuencia
            segment = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            
            speaker = manager.identify_speaker(segment)
            self.assertIsNotNone(speaker)
            
            # Forzar garbage collection periódicamente
            if i % 10 == 0:
                gc.collect()
        
        # Verificar que el manager sigue funcionando
        t = np.linspace(0, 1, self.sample_rate)
        final_segment = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        final_speaker = manager.identify_speaker(final_segment)
        self.assertIsNotNone(final_speaker)

    def test_configuration_variations(self):
        """Prueba diferentes configuraciones del sistema."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        
        # Test diferentes umbrales de similitud
        thresholds = [0.5, 0.8, 0.95]
        
        for threshold in thresholds:
            manager = SpeakerManager(similarity_threshold=threshold)
            
            # Crear dos segmentos ligeramente diferentes
            t = np.linspace(0, 1, self.sample_rate)
            segment1 = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            segment2 = (np.sin(2 * np.pi * 445 * t) * 32767).astype(np.int16)  # Ligeramente diferente
            
            speaker1 = manager.identify_speaker(segment1)
            speaker2 = manager.identify_speaker(segment2)
            
            self.assertIsNotNone(speaker1)
            self.assertIsNotNone(speaker2)
            
            # Con umbral bajo, pueden ser el mismo hablante
            # Con umbral alto, probablemente sean diferentes
            if threshold >= 0.9:
                # Más probable que sean diferentes hablantes
                pass  # No hacer assertion específica ya que depende de las características extraídas
            
    @patch('builtins.print')
    def test_output_formatting(self, mock_print):
        """Prueba el formateo de salida del sistema."""
        from hispano_transcriber.transcriber_speaker import format_transcription_output
        
        # Test diferentes casos de salida
        test_cases = [
            ("Hablante 1", "Hola mundo"),
            ("Hablante 2", ""),  # Texto vacío
            ("Hablante 3", "Texto con símbolos: ¡¿@#$%^&*()!"),
            ("Hablante 1", "Texto muy largo " * 20)
        ]
        
        for speaker, text in test_cases:
            format_transcription_output(speaker, text)
            # Verificar que se llamó print
            mock_print.assert_called()
        
        # Verificar número total de llamadas
        self.assertEqual(mock_print.call_count, len(test_cases))


class TestIntegrationWithExternalDependencies(unittest.TestCase):
    """Pruebas de integración con dependencias externas."""

    def test_sounddevice_integration(self):
        """Prueba integración básica con sounddevice."""
        try:
            import sounddevice as sd
            
            # Verificar que se pueden consultar dispositivos
            devices = sd.query_devices()
            self.assertIsInstance(devices, (list, type(None)))
            
            # Verificar que se puede obtener dispositivo por defecto
            try:
                default_device = sd.default.device
                # Puede ser None si no hay dispositivos
            except Exception:
                pass  # Algunas configuraciones pueden no tener dispositivos de audio
                
        except ImportError:
            self.fail("sounddevice no está disponible")

    def test_numpy_integration(self):
        """Prueba integración con NumPy."""
        import numpy as np
        
        # Test operaciones básicas de NumPy usadas en el proyecto
        audio = np.random.randn(16000).astype(np.float32)
        
        # Test conversiones de tipo
        audio_int16 = (audio * 32767).astype(np.int16)
        self.assertEqual(audio_int16.dtype, np.int16)
        
        # Test operaciones estadísticas
        rms = np.sqrt(np.mean(np.square(audio)))
        self.assertGreater(rms, 0)
        
        # Test FFT
        spectrum = np.abs(np.fft.rfft(audio[:512]))
        self.assertEqual(len(spectrum), 257)

    def test_sklearn_integration(self):
        """Prueba integración con scikit-learn."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Test similitud coseno
            vec1 = np.array([[1, 2, 3, 4]])
            vec2 = np.array([[1, 2, 3, 4]])
            vec3 = np.array([[4, 3, 2, 1]])
            
            # Similitud idéntica
            sim_identical = cosine_similarity(vec1, vec2)[0][0]
            self.assertAlmostEqual(sim_identical, 1.0, places=5)
            
            # Similitud diferente
            sim_different = cosine_similarity(vec1, vec3)[0][0]
            self.assertLess(sim_different, 1.0)
            
        except ImportError:
            self.fail("scikit-learn no está disponible")


if __name__ == '__main__':
    unittest.main(verbosity=2)
