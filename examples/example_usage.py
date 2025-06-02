"""
Ejemplo de uso del transcriptor hispano con identificación de hablantes.

Este script muestra cómo utilizar las principales funciones del transcriptor.
"""

from hispano_transcriber import transcriber_speaker

def main():
    """Ejemplo de uso del transcriptor con identificación de hablantes"""
    print("=== Ejemplo de uso del transcriptor hispano ===")
    print("Este ejemplo ejecutará el transcriptor con identificación de hablantes.")
    print("Hable en español cerca de su micrófono para ver los resultados.")
    print("Presione Ctrl+C para finalizar.\n")
    
    # Ejecutar el transcriptor con un umbral de similitud personalizado
    transcriber_speaker.main()

if __name__ == "__main__":
    main()
