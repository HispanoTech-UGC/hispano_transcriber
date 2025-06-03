"""
Tests para Hispano Transcriber.
"""

import sys
import os

# Agregar el directorio src al path para poder importar hispano_transcriber
tests_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(tests_dir)
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
