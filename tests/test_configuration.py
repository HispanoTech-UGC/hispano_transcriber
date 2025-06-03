"""
Tests de configuración y utilidades para el proyecto hispano_transcriber.
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch
import tempfile
import shutil

# Agregar el directorio src al path para poder importar hispano_transcriber
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


class TestProjectConfiguration(unittest.TestCase):
    """Pruebas de configuración del proyecto."""

    def test_imports_availability(self):
        """Prueba que todas las dependencias necesarias están disponibles."""
        try:
            import sounddevice
            import vosk
            import numpy
            import transformers
            import torch
            import sklearn
        except ImportError as e:
            self.fail(f"Dependencia faltante: {e}")

    def test_module_imports(self):
        """Prueba que los módulos del proyecto se pueden importar."""
        try:
            from hispano_transcriber import transcriber
            from hispano_transcriber import transcriber_speaker
        except ImportError as e:
            self.fail(f"No se pudo importar módulo del proyecto: {e}")

    def test_vosk_model_path_format(self):
        """Prueba que la ruta del modelo Vosk tiene el formato correcto."""
        from hispano_transcriber import transcriber
        from hispano_transcriber import transcriber_speaker
        
        # Verificar que las rutas tienen formato esperado
        self.assertIn("vosk-model-es", transcriber.MODEL_PATH)
        self.assertIn("vosk-model-es", transcriber_speaker.MODEL_PATH)

    def test_audio_parameters_consistency(self):
        """Prueba que los parámetros de audio son consistentes entre módulos."""
        from hispano_transcriber import transcriber
        from hispano_transcriber import transcriber_speaker
        
        # Verificar consistencia de parámetros de audio
        self.assertEqual(transcriber.SAMPLE_RATE, transcriber_speaker.SAMPLE_RATE)
        self.assertEqual(transcriber.BLOCK_SIZE, transcriber_speaker.BLOCK_SIZE)


class TestProjectStructure(unittest.TestCase):
    """Pruebas de estructura del proyecto."""

    def setUp(self):
        """Preparación para las pruebas."""
        # Obtener el directorio raíz del proyecto
        current_dir = os.path.dirname(__file__)
        self.project_root = os.path.dirname(current_dir)

    def test_required_files_exist(self):
        """Prueba que existen los archivos requeridos del proyecto."""
        required_files = [
            'README.md',
            'LICENSE',
            'setup.py',
            'requirements.txt',
            'requirements-dev.txt'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(self.project_root, file_name)
            self.assertTrue(os.path.exists(file_path), f"Archivo faltante: {file_name}")

    def test_source_directory_structure(self):
        """Prueba la estructura del directorio de código fuente."""
        src_dir = os.path.join(self.project_root, 'src', 'hispano_transcriber')
        
        required_files = [
            '__init__.py',
            'transcriber.py',
            'transcriber_speaker.py'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(src_dir, file_name)
            self.assertTrue(os.path.exists(file_path), f"Archivo fuente faltante: {file_name}")

    def test_tests_directory_structure(self):
        """Prueba la estructura del directorio de tests."""
        tests_dir = os.path.join(self.project_root, 'tests')
        
        self.assertTrue(os.path.exists(tests_dir), "Directorio tests no existe")
        
        init_file = os.path.join(tests_dir, '__init__.py')
        self.assertTrue(os.path.exists(init_file), "Archivo __init__.py faltante en tests")

    def test_examples_directory_exists(self):
        """Prueba que existe el directorio de ejemplos."""
        examples_dir = os.path.join(self.project_root, 'examples')
        self.assertTrue(os.path.exists(examples_dir), "Directorio examples no existe")


class TestUtilityFunctions(unittest.TestCase):
    """Pruebas de funciones utilitarias."""

    def test_audio_data_conversion(self):
        """Prueba conversión de datos de audio."""
        import numpy as np
        
        # Crear datos de audio sintéticos
        sample_rate = 16000
        duration = 1  # segundo
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio_float = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Convertir a int16 (formato usado por el transcriptor)
        audio_int16 = (audio_float * 32767).astype(np.int16)
        
        # Verificar rango válido
        self.assertTrue(np.all(audio_int16 >= -32768))
        self.assertTrue(np.all(audio_int16 <= 32767))
        
        # Verificar que no hay overflow
        self.assertEqual(audio_int16.dtype, np.int16)

    def test_json_parsing_robustness(self):
        """Prueba robustez del parsing JSON."""
        import json
        
        # Test casos válidos
        valid_json = '{"text": "Hola mundo", "confidence": 0.95}'
        result = json.loads(valid_json)
        self.assertEqual(result["text"], "Hola mundo")
        
        # Test casos inválidos
        invalid_jsons = [
            '{"text": "Incompleto"',  # JSON incompleto
            '{"text":}',  # Valor faltante
            '',  # Cadena vacía
            'not_json_at_all'  # No es JSON
        ]
        
        for invalid_json in invalid_jsons:
            with self.assertRaises(json.JSONDecodeError):
                json.loads(invalid_json)

    def test_numpy_array_operations(self):
        """Prueba operaciones con arrays de NumPy."""
        import numpy as np
        
        # Crear array de prueba
        audio = np.random.randn(16000).astype(np.float32)
        
        # Test operaciones comunes
        rms = np.sqrt(np.mean(np.square(audio)))
        self.assertGreaterEqual(rms, 0)
        
        # Test zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        zcr = zero_crossings / len(audio)
        self.assertGreaterEqual(zcr, 0)
        self.assertLessEqual(zcr, 1)
        
        # Test FFT
        if len(audio) >= 512:
            spectrum = np.abs(np.fft.rfft(audio[:512]))
            self.assertEqual(len(spectrum), 257)  # 512/2 + 1


class TestErrorHandling(unittest.TestCase):
    """Pruebas de manejo de errores."""

    def test_missing_model_error_handling(self):
        """Prueba manejo de errores cuando falta el modelo."""
        from hispano_transcriber import transcriber
        
        with patch('hispano_transcriber.transcriber.os.path.exists', return_value=False):
            with patch('sys.exit') as mock_exit:
                transcriber.main()
                mock_exit.assert_called_once_with(1)

    def test_audio_device_error_simulation(self):
        """Prueba simulación de errores de dispositivo de audio."""
        from hispano_transcriber import transcriber
        import numpy as np
        
        # Simular error en callback
        indata = np.zeros((3200, 1), dtype=np.int16)
        frames = 3200
        time_info = None
        status = "Device error"
        
        with patch('hispano_transcriber.transcriber.audio_queue') as mock_queue:
            with patch('sys.stderr'):
                # No debería lanzar excepción
                transcriber.callback(indata, frames, time_info, status)
                mock_queue.put.assert_called_once()

    def test_memory_allocation_error_simulation(self):
        """Prueba simulación de errores de memoria."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        import numpy as np
        
        manager = SpeakerManager()
        
        # Simular array muy grande que podría causar error de memoria
        try:
            # En lugar de crear un array gigante, simulamos el comportamiento
            large_audio = np.zeros(1000, dtype=np.int16)  # Pequeño para la prueba
            features = manager.extract_voice_features(large_audio)
            # Si llega aquí, la función manejó correctamente el caso
            self.assertIsNotNone(features)
        except MemoryError:
            # Si hay error de memoria, debería manejarse graciosamente
            pass


class TestPerformanceBasics(unittest.TestCase):
    """Pruebas básicas de rendimiento."""

    def test_feature_extraction_performance(self):
        """Prueba básica de rendimiento de extracción de características."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        import numpy as np
        import time
        
        manager = SpeakerManager()
        
        # Crear audio de prueba
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        # Medir tiempo de extracción
        start_time = time.time()
        features = manager.extract_voice_features(audio)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # La extracción debería ser rápida (menos de 1 segundo para 2 segundos de audio)
        self.assertLess(execution_time, 1.0)
        self.assertIsNotNone(features)

    def test_speaker_identification_performance(self):
        """Prueba básica de rendimiento de identificación de hablantes."""
        from hispano_transcriber.transcriber_speaker import SpeakerManager
        import numpy as np
        import time
        
        manager = SpeakerManager()
        
        # Crear audio de prueba
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        # Medir tiempo de identificación
        start_time = time.time()
        speaker_id = manager.identify_speaker(audio)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # La identificación debería ser rápida
        self.assertLess(execution_time, 2.0)
        self.assertIsNotNone(speaker_id)
        self.assertTrue(speaker_id.startswith("Hablante"))


if __name__ == '__main__':
    unittest.main()
