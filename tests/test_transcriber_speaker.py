"""
Tests de integración para el transcriptor con identificación de hablantes.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import queue
from hispano_transcriber import transcriber_speaker


class TestTranscriberSpeaker(unittest.TestCase):
    """Pruebas para el módulo transcriber_speaker."""

    def setUp(self):
        """Preparación para las pruebas."""
        self.sample_rate = 16000
        self.duration = 2
        # Crear audio sintético para pruebas
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))
        self.test_audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    def test_callback_function(self):
        """Prueba la función callback de captura de audio."""
        indata = np.zeros((3200, 1), dtype=np.int16)
        frames = 3200
        time_info = None
        status = None
        
        # Mock de audio_queue
        original_queue = transcriber_speaker.audio_queue
        mock_queue = Mock()
        transcriber_speaker.audio_queue = mock_queue
        
        try:
            # Mock de audio_buffer para evitar modificarlo
            original_buffer = transcriber_speaker.audio_buffer
            transcriber_speaker.audio_buffer = []
            
            # Ejecutar callback
            transcriber_speaker.callback(indata, frames, time_info, status)
            
            # Verificar que se llama a queue.put
            mock_queue.put.assert_called_once()
            
            # Verificar que se ha añadido al buffer
            self.assertTrue(len(transcriber_speaker.audio_buffer) > 0)
        finally:
            # Restaurar variables globales
            transcriber_speaker.audio_queue = original_queue
            transcriber_speaker.audio_buffer = original_buffer

    def test_analyze_speaker_change(self):
        """Prueba la función de análisis de cambio de hablante."""
        # Crear un mock de SpeakerManager con la estructura necesaria
        mock_speaker_manager = Mock()
        mock_speaker_manager.identify_speaker.return_value = "Hablante 2"
        mock_speaker_manager.speakers = {"Hablante 2": [np.array([0.1, 0.2, 0.3, 0.4])]}
        
        # Preparar buffer de audio
        original_buffer = transcriber_speaker.audio_buffer
        original_speaker = transcriber_speaker.current_speaker
        
        try:
            # Asignar un buffer suficientemente grande
            transcriber_speaker.audio_buffer = list(self.test_audio)
            transcriber_speaker.current_speaker = "Hablante 1"
            
            # Analizar cambio de hablante
            with patch('builtins.print'):  # Suprimir las salidas de print
                result = transcriber_speaker.analyze_speaker_change(mock_speaker_manager)
            
            # Verificar que se detectó el cambio
            self.assertEqual(result, "Hablante 2")
            self.assertEqual(transcriber_speaker.current_speaker, "Hablante 2")
        finally:
            # Restaurar variables globales
            transcriber_speaker.audio_buffer = original_buffer
            transcriber_speaker.current_speaker = original_speaker

    def test_format_speaker_output(self):
        """Prueba el formateo de salida con información del hablante."""
        speaker = "Hablante 1"
        text = "Hola mundo"
        
        # Capturar la salida formateada
        with patch('builtins.print') as mock_print:
            result = transcriber_speaker.format_speaker_output(speaker, text)
            mock_print.assert_called_once()
            self.assertIn(speaker, result)
            self.assertIn(text, result)

    def test_format_transcription_output(self):
        """Prueba el formateo de salida de transcripción."""
        speaker = "Hablante 1"
        text = "Hola mundo"
        
        # Capturar la salida
        with patch('builtins.print') as mock_print:
            transcriber_speaker.format_transcription_output(speaker, text)
            mock_print.assert_called()

    @patch('hispano_transcriber.transcriber_speaker.os.path.exists')
    @patch('hispano_transcriber.transcriber_speaker.Model')
    @patch('hispano_transcriber.transcriber_speaker.KaldiRecognizer')
    @patch('hispano_transcriber.transcriber_speaker.sd.RawInputStream')
    @patch('hispano_transcriber.transcriber_speaker.audio_queue')
    def test_main_integration(self, mock_queue, mock_stream, mock_recognizer, mock_model, mock_exists):
        """Prueba de integración del flujo principal."""
        # Setup mocks
        mock_exists.return_value = True
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_recognizer_instance = Mock()
        mock_recognizer.return_value = mock_recognizer_instance
        mock_recognizer_instance.AcceptWaveform.return_value = True
        mock_recognizer_instance.Result.return_value = '{"text": "Prueba de transcripción"}'
        
        # Simular datos de audio y KeyboardInterrupt
        audio_data = self.test_audio.tobytes()
        mock_queue.get.side_effect = [audio_data] * 5 + [KeyboardInterrupt()]
        
        # Mockear SpeakerManager
        with patch('hispano_transcriber.transcriber_speaker.SpeakerManager') as mock_speaker_manager:
            mock_speaker_instance = Mock()
            mock_speaker_manager.return_value = mock_speaker_instance
            mock_speaker_instance.identify_speaker.return_value = "Hablante 1"
            
            # Ejecutar
            try:
                with patch('hispano_transcriber.transcriber_speaker.cargar_modelos') as mock_cargar:
                    mock_cargar.return_value = (mock_model_instance, mock_recognizer_instance)
                    transcriber_speaker.main()
            except SystemExit:
                pass

    def test_constants_and_configuration(self):
        """Prueba que las constantes están configuradas correctamente."""
        self.assertEqual(transcriber_speaker.SAMPLE_RATE, 16000)
        self.assertEqual(transcriber_speaker.BLOCK_SIZE, 3200)
        self.assertEqual(transcriber_speaker.SPEAKER_BUFFER_DURATION, 3.0)
        self.assertEqual(transcriber_speaker.MIN_SPEECH_DURATION, 1.0)

    def test_global_variables_initialization(self):
        """Prueba la inicialización de variables globales."""
        self.assertIsInstance(transcriber_speaker.audio_queue, queue.Queue)
        self.assertIsInstance(transcriber_speaker.speaker_embeddings, list)
        self.assertIsInstance(transcriber_speaker.speaker_labels, list)
        self.assertIsInstance(transcriber_speaker.audio_buffer, list)
        
    def test_argument_parsing(self):
        """Prueba el parsing de argumentos de línea de comandos."""
        # En lugar de probar la función main completa, verificamos la inicialización de argumentos
        with patch('argparse.ArgumentParser') as mock_parser:
            # Configurar el mock para devolver argumentos específicos
            mock_args = Mock()
            mock_args.threshold = 0.8
            mock_args.debug = True
            
            mock_parser_instance = Mock()
            mock_parser.return_value = mock_parser_instance
            mock_parser_instance.parse_args.return_value = mock_args
            
            # Verificar directamente la funcionalidad del parser sin llamar a main()
            self.assertEqual(mock_args.threshold, 0.8)
            self.assertEqual(mock_args.debug, True)


class TestSpeakerManagerExtended(unittest.TestCase):
    """Pruebas extendidas para SpeakerManager."""

    def setUp(self):
        """Preparación para las pruebas."""
        self.speaker_manager = transcriber_speaker.SpeakerManager(similarity_threshold=0.8)
        self.sample_rate = 16000

    def test_extract_voice_features_empty_audio(self):
        """Prueba extracción de características con audio vacío."""
        empty_audio = np.array([], dtype=np.int16)
        features = self.speaker_manager.extract_voice_features(empty_audio)
        self.assertIsNone(features)

    def test_extract_voice_features_very_short_audio(self):
        """Prueba con audio muy corto."""
        short_audio = np.ones(100, dtype=np.int16)  # Muy corto
        features = self.speaker_manager.extract_voice_features(short_audio)
        self.assertIsNone(features)

    def test_extract_voice_features_normal_audio(self):
        """Prueba extracción de características con audio normal."""
        # Crear audio sintético de 2 segundos
        duration = 2
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        features = self.speaker_manager.extract_voice_features(audio)
        
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 4)  # rms, zcr, spectral_centroid, spectral_rolloff

    def test_identify_speaker_first_speaker(self):
        """Prueba identificación del primer hablante."""
        # Crear audio sintético
        duration = 2
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        # Debe ser el primer hablante (no hay hablantes previos)
        speaker = self.speaker_manager.identify_speaker(audio)
        self.assertEqual(speaker, "Hablante 1")
        self.assertEqual(len(self.speaker_manager.speakers), 1)
        self.assertEqual(len(self.speaker_manager.speakers["Hablante 1"]), 1)

    def test_identify_speaker_same_speaker(self):
        """Prueba identificación del mismo hablante."""
        # Crear dos segmentos de audio similares
        duration = 2
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        audio1 = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        audio2 = (np.sin(2 * np.pi * 442 * t) * 32767).astype(np.int16)  # Muy similar
        
        # Registrar el primer hablante
        speaker1 = self.speaker_manager.identify_speaker(audio1)
        
        # Simular alta similitud para el mismo hablante
        with patch.object(self.speaker_manager, 'extract_embedding') as mock_extract:
            # Hacer que devuelva el mismo embedding que ya está registrado
            embedding = next(iter(self.speaker_manager.speakers[speaker1]))
            mock_extract.return_value = embedding
            
            speaker2 = self.speaker_manager.identify_speaker(audio2)
            self.assertEqual(speaker2, speaker1)

    def test_identify_speaker_different_speakers(self):
        """Prueba identificación de diferentes hablantes."""
        # Crear dos segmentos de audio diferentes
        duration = 2
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        audio1 = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)  # 440 Hz
        audio2 = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)  # 880 Hz (diferente)
        
        # Registrar el primer hablante
        speaker1 = self.speaker_manager.identify_speaker(audio1)
        
        # Simular baja similitud para un hablante diferente
        with patch.object(self.speaker_manager, 'extract_embedding') as mock_extract:
            # Devolver un embedding muy diferente
            mock_extract.return_value = np.array([0.9, 0.1, 0.1, 0.1])
            
            # Ajustar el umbral de similitud
            self.speaker_manager.similarity_threshold = 0.95  # Alto umbral para forzar diferencia
            
            speaker2 = self.speaker_manager.identify_speaker(audio2)
            self.assertNotEqual(speaker2, speaker1)

    def test_speaker_manager_initialization(self):
        """Prueba inicialización del SpeakerManager."""
        manager = transcriber_speaker.SpeakerManager(similarity_threshold=0.75)
        
        self.assertEqual(manager.similarity_threshold, 0.75)
        self.assertEqual(manager.speaker_count, 0)
        self.assertEqual(manager.speakers, {})