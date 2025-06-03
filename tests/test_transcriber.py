"""
Tests unitarios para el módulo transcriber.py
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import queue
import numpy as np
from hispano_transcriber import transcriber


class TestTranscriber(unittest.TestCase):
    """Pruebas para el módulo transcriber."""

    def setUp(self):
        """Preparación para las pruebas."""
        self.sample_rate = 16000
        self.block_size = 3200

    @patch('hispano_transcriber.transcriber.os.path.exists')
    @patch('hispano_transcriber.transcriber.Model')
    @patch('hispano_transcriber.transcriber.KaldiRecognizer')
    @patch('hispano_transcriber.transcriber.sd.RawInputStream')
    @patch('hispano_transcriber.transcriber.audio_queue')
    def test_main_model_not_found(self, mock_queue, mock_stream, mock_recognizer, mock_model, mock_exists):
        """Prueba que el programa termina cuando no se encuentra el modelo."""
        mock_exists.return_value = False
        
        # Añadir esto para evitar que audio_queue.get() bloquee el test
        mock_queue.get.side_effect = KeyboardInterrupt()
        
        with patch('sys.exit') as mock_exit:
            transcriber.main()
            mock_exit.assert_called_once_with(1)

    @patch('hispano_transcriber.transcriber.os.path.exists')
    @patch('hispano_transcriber.transcriber.Model')
    @patch('hispano_transcriber.transcriber.KaldiRecognizer')
    @patch('hispano_transcriber.transcriber.sd.RawInputStream')
    @patch('hispano_transcriber.transcriber.audio_queue')
    def test_main_successful_transcription(self, mock_queue, mock_stream, mock_recognizer, mock_model, mock_exists):
        """Prueba una transcripción exitosa."""
        # Setup mocks
        mock_exists.return_value = True
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_recognizer_instance = Mock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Simular resultado de reconocimiento
        mock_recognizer_instance.AcceptWaveform.side_effect = [True, False]
        mock_recognizer_instance.Result.return_value = '{"text": "Hola mundo"}'
        mock_recognizer_instance.PartialResult.return_value = '{"partial": "Hola"}'
        
        # Simular datos de audio
        audio_data = b'fake_audio_data'
        mock_queue.get.side_effect = [audio_data, KeyboardInterrupt()]
        
        # Ejecutar y verificar que no hay errores
        try:
            transcriber.main()
        except SystemExit:
            pass  # Esperado cuando se simula Ctrl+C

    def test_callback_with_status_error(self):
        """Prueba el callback con error de estado."""
        indata = np.zeros((3200, 1), dtype=np.int16)
        frames = 3200
        time_info = None
        status = "Error de prueba"
        
        with patch('hispano_transcriber.transcriber.audio_queue') as mock_queue:
            with patch('sys.stderr') as mock_stderr:
                transcriber.callback(indata, frames, time_info, status)
                mock_queue.put.assert_called_once()

    def test_callback_normal_operation(self):
        """Prueba el callback en operación normal."""
        indata = np.zeros((3200, 1), dtype=np.int16)
        frames = 3200
        time_info = None
        status = None
        
        with patch('hispano_transcriber.transcriber.audio_queue') as mock_queue:
            transcriber.callback(indata, frames, time_info, status)
            mock_queue.put.assert_called_once_with(bytes(indata))

    @patch('hispano_transcriber.transcriber.os.path.exists')
    @patch('hispano_transcriber.transcriber.Model')
    @patch('hispano_transcriber.transcriber.KaldiRecognizer')
    @patch('hispano_transcriber.transcriber.sd.RawInputStream')
    @patch('hispano_transcriber.transcriber.audio_queue')
    def test_main_json_parsing_error(self, mock_queue, mock_stream, mock_recognizer, mock_model, mock_exists):
        """Prueba el manejo de errores de parsing JSON."""
        # Setup mocks
        mock_exists.return_value = True
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_recognizer_instance = Mock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Simular JSON malformado
        mock_recognizer_instance.AcceptWaveform.return_value = True
        mock_recognizer_instance.Result.return_value = 'invalid_json'
        
        # Simular datos de audio y luego KeyboardInterrupt
        audio_data = b'fake_audio_data'
        mock_queue.get.side_effect = [audio_data, KeyboardInterrupt()]
        
        # Ejecutar y verificar que maneja el error
        try:
            transcriber.main()
        except SystemExit:
            pass

    @patch('hispano_transcriber.transcriber.os.path.exists')
    @patch('hispano_transcriber.transcriber.Model')
    @patch('hispano_transcriber.transcriber.KaldiRecognizer')
    @patch('hispano_transcriber.transcriber.sd.RawInputStream')
    @patch('hispano_transcriber.transcriber.audio_queue')
    def test_main_partial_result_processing(self, mock_queue, mock_stream, mock_recognizer, mock_model, mock_exists):
        """Prueba el procesamiento de resultados parciales."""
        # Setup mocks
        mock_exists.return_value = True
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_recognizer_instance = Mock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Simular resultado parcial
        mock_recognizer_instance.AcceptWaveform.return_value = False
        mock_recognizer_instance.PartialResult.return_value = '{"partial": "Hablando..."}'
        
        # Simular datos de audio
        audio_data = b'fake_audio_data'
        mock_queue.get.side_effect = [audio_data, KeyboardInterrupt()]
        
        # Ejecutar
        try:
            transcriber.main()
        except SystemExit:
            pass

    def test_constants_configuration(self):
        """Prueba que las constantes están configuradas correctamente."""
        self.assertEqual(transcriber.SAMPLE_RATE, 16000)
        self.assertEqual(transcriber.BLOCK_SIZE, 3200)
        self.assertTrue(transcriber.MODEL_PATH.endswith("vosk-model-es-0.42"))

    def test_audio_queue_initialization(self):
        """Prueba que la cola de audio se inicializa correctamente."""
        self.assertIsInstance(transcriber.audio_queue, queue.Queue)


if __name__ == '__main__':
    unittest.main()