"""
Punto de entrada para ejecutar el paquete directamente.
"""

import sys
from .transcriber_speaker import main as speaker_main
from .transcriber import main as simple_main


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        # Modo simple sin identificación de hablantes
        simple_main()
    else:
        # Modo con identificación de hablantes por defecto
        speaker_main()
