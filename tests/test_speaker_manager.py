"""
Tests unitarios para el módulo de identificación de hablantes.
"""

import unittest
import numpy as np
from hispano_transcriber.transcriber_speaker import SpeakerManager


class TestSpeakerManager(unittest.TestCase):
    """Pruebas para la clase SpeakerManager."""
    
    def setUp(self):
        """Preparación para las pruebas."""
        self.speaker_manager = SpeakerManager(similarity_threshold=0.8)
    
    def test_extract_voice_features(self):
        """Prueba la extracción de características de voz."""
        # Crear un segmento de audio sintético
        sample_rate = 16000
        duration = 2  # segundos
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 32767  # Tono A (440 Hz)
        
        # Extraer características
        features = self.speaker_manager.extract_voice_features(audio)
        
        # Verificar que se obtuvieron características
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 4)  # rms, zcr, spectral_centroid, spectral_rolloff
    
    def test_identify_speaker_new_speaker(self):
        """Prueba la identificación cuando no hay hablantes previos."""
        # Crear un segmento de audio sintético
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 32767
        
        # Identificar hablante (debería ser el primer hablante)
        speaker_id = self.speaker_manager.identify_speaker(audio)
        self.assertEqual(speaker_id, "Hablante 1")
        
    def test_identify_speaker_same_speaker(self):
        """Prueba la identificación cuando el mismo hablante aparece dos veces."""
        # Crear dos segmentos de audio similares (mismo hablante)
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 32767
        audio2 = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 32767 * 0.95  # Ligeramente diferente
        
        # Mockear la función extract_embedding para que devuelva embeddings muy similares
        original_extract_embedding = self.speaker_manager.extract_embedding
        
        # Primer embedding
        embedding1 = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Segundo embedding (muy similar al primero)
        embedding2 = np.array([0.11, 0.19, 0.31, 0.39])
        
        embeddings = [embedding1, embedding2]
        
        def mock_extract_embedding(audio):
            return embeddings.pop(0) if embeddings else None
        
        self.speaker_manager.extract_embedding = mock_extract_embedding
        
        try:
            # Identificar primer hablante
            speaker1 = self.speaker_manager.identify_speaker(audio1)
            # Identificar segundo hablante (debería ser el mismo)
            speaker2 = self.speaker_manager.identify_speaker(audio2)
            
            self.assertEqual(speaker1, speaker2)
        finally:
            # Restaurar la función original
            self.speaker_manager.extract_embedding = original_extract_embedding
    
    def test_identify_speaker_different_speakers(self):
        """Prueba la identificación cuando hay diferentes hablantes."""
        # Mockear la función extract_embedding para que devuelva embeddings muy diferentes
        original_extract_embedding = self.speaker_manager.extract_embedding
        
        # Primer embedding
        embedding1 = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Segundo embedding (muy diferente del primero)
        embedding2 = np.array([0.8, 0.7, 0.6, 0.5])
        
        embeddings = [embedding1, embedding2]
        
        def mock_extract_embedding(audio):
            return embeddings.pop(0) if embeddings else None
        
        self.speaker_manager.extract_embedding = mock_extract_embedding
        
        try:
            # Identificar primer hablante
            speaker1 = self.speaker_manager.identify_speaker(np.zeros(16000))
            # Identificar segundo hablante (debería ser diferente)
            speaker2 = self.speaker_manager.identify_speaker(np.zeros(16000))
            
            self.assertNotEqual(speaker1, speaker2)
        finally:
            # Restaurar la función original
            self.speaker_manager.extract_embedding = original_extract_embedding


if __name__ == '__main__':
    unittest.main()
