# Documentación de Tests - Hispano Transcriber

Este documento describe los tests implementados para el proyecto Hispano Transcriber, organizados por categorías.

## Estructura de Tests

### Pruebas Unitarias

- **test_transcriber.py**
  - Comprueba: El funcionamiento básico del transcriptor de voz sin identificación de hablantes.
  - Verifica el callback de audio, la inicialización de la cola, y la gestión de errores.

- **test_speaker_manager.py**
  - Comprueba: La clase SpeakerManager que gestiona la identificación de hablantes.
  - Verifica la extracción de características de voz y la correcta identificación de diferentes hablantes.

- **test_configuration.py**
  - Comprueba: La correcta configuración del proyecto y la disponibilidad de dependencias.
  - Verifica la estructura del proyecto y la capacidad de importación de los módulos necesarios.

### Pruebas de Integración

- **test_transcriber_speaker.py**
  - Comprueba: La integración del transcriptor de voz con la identificación de hablantes.
  - Verifica el procesamiento de audio, la detección de cambios de hablante y el formateado de salida.

- **test_end_to_end.py**
  - Comprueba: El flujo de trabajo completo desde la captura de audio hasta la transcripción final.
  - Verifica el pipeline completo y la integración con dependencias externas.

### Pruebas de Rendimiento

- **test_performance.py**
  - Comprueba: El rendimiento del sistema con diferentes configuraciones y cargas.
  - Verifica la latencia, el uso de memoria y la capacidad de procesamiento concurrente.

### Pruebas de Casos Límite

- **test_edge_cases.py**
  - Comprueba: El comportamiento del sistema con datos de entrada no estándar o extremos.
  - Verifica la robustez frente a audio silencioso, con ruido, corrupto, o de duración extrema.

## Detalles de Tests Específicos

### test_transcriber.py

- **test_callback_normal_operation**
  - Comprueba: Que el callback procesa correctamente los datos de audio en operación normal.

- **test_main_model_not_found**
  - Comprueba: Que el sistema maneja correctamente la ausencia del modelo de Vosk.

- **test_main_successful_transcription**
  - Comprueba: La correcta transcripción de audio con resultados completos.

- **test_main_json_parsing_error**
  - Comprueba: El manejo adecuado de errores en el formato JSON.

- **test_main_partial_result_processing**
  - Comprueba: El procesamiento de resultados parciales de transcripción.

### test_speaker_manager.py

- **test_extract_voice_features**
  - Comprueba: La extracción de características acústicas del audio.

- **test_identify_speaker_new_speaker**
  - Comprueba: La identificación correcta de un nuevo hablante.

- **test_identify_speaker_same_speaker**
  - Comprueba: La identificación correcta del mismo hablante en diferentes segmentos.

- **test_identify_speaker_different_speakers**
  - Comprueba: La diferenciación correcta entre hablantes distintos.

### test_configuration.py

- **test_imports_availability**
  - Comprueba: La disponibilidad de todas las dependencias requeridas.

- **test_required_files_exist**
  - Comprueba: La existencia de archivos necesarios para el funcionamiento.

- **test_source_directory_structure**
  - Comprueba: La correcta estructura de directorios del código fuente.

### test_transcriber_speaker.py

- **test_callback_function**
  - Comprueba: El funcionamiento del callback que procesa audio y lo añade al buffer.

- **test_analyze_speaker_change**
  - Comprueba: La detección correcta de cambios de hablante.

- **test_format_speaker_output**
  - Comprueba: El formateado correcto de la salida con identificación de hablante.

- **test_argument_parsing**
  - Comprueba: El correcto análisis de argumentos de línea de comandos.

### test_end_to_end.py

- **test_complete_transcription_workflow**
  - Comprueba: El flujo completo del sistema sin identificación de hablantes.

- **test_complete_speaker_transcription_workflow**
  - Comprueba: El flujo completo con identificación de hablantes.

- **test_output_formatting**
  - Comprueba: El formato correcto de la salida final al usuario.

### test_performance.py

- **test_transcription_latency**
  - Comprueba: La latencia del proceso de transcripción.

- **test_throughput_multiple_speakers**
  - Comprueba: La capacidad de procesamiento con múltiples hablantes.

- **test_memory_usage_over_time**
  - Comprueba: El uso de memoria durante periodos largos de transcripción.

### test_edge_cases.py

- **test_silent_audio_handling**
  - Comprueba: El manejo de segmentos de audio silenciosos.

- **test_very_short_audio_segments**
  - Comprueba: El comportamiento con segmentos de audio muy cortos.

- **test_very_long_audio_segments**
  - Comprueba: La capacidad de procesar segmentos de audio muy largos.

- **test_maximum_number_of_speakers**
  - Comprueba: El manejo de un gran número de hablantes diferentes.

## Ejecución de Tests

Los tests pueden ejecutarse utilizando pytest:

```bash
# Ejecutar todos los tests
pytest tests/

# Ejecutar una categoría específica
pytest tests/test_transcriber.py

# Ejecutar un test específico
pytest tests/test_transcriber.py::TestTranscriber::test_main_successful_transcription

# Ejecutar con indicadores de cobertura
pytest tests/ --cov=hispano_transcriber
```

## Dependencias de Tests

- pytest
- pytest-cov (para análisis de cobertura)
- pytest-timeout (para evitar bloqueos en pruebas)
- NumPy (para generar datos de prueba)
- unittest.mock (para simular comportamientos y dependencias)
