# Hispano Transcriber

## Documentación de Uso

### Instalación

Para instalar Hispano Transcriber, sigue estos pasos:

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/hispano-transcriber.git
cd hispano-transcriber

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

### Uso Básico

#### Transcripción Simple

Para usar el transcriptor básico sin identificación de hablantes:

```python
from hispano_transcriber import transcriber

# Ejecutar el transcriptor
transcriber.main()
```

#### Transcripción con Identificación de Hablantes

Para usar el transcriptor con identificación automática de hablantes:

```python
from hispano_transcriber import transcriber_speaker

# Ejecutar el transcriptor con identificación de hablantes
transcriber_speaker.main()
```

### Configuración Avanzada

Puedes ajustar el umbral de similitud para la identificación de hablantes:

```python
# Desde la línea de comandos
python -m hispano_transcriber.transcriber_speaker --umbral 0.75 --simple
```

En este ejemplo:
- `--umbral 0.75` establece un umbral más bajo (más permisivo) para considerar que dos segmentos de audio pertenecen al mismo hablante.
- `--simple` indica que se deben usar características básicas de audio en lugar del modelo Wav2Vec2.

### Integración en Proyectos

Para integrar el transcriptor en tu propio proyecto:

```python
import time
from hispano_transcriber.transcriber_speaker import SpeakerManager, cargar_modelos

# Cargar modelos
model, recognizer = cargar_modelos()

# Inicializar gestor de hablantes con umbral personalizado
speaker_manager = SpeakerManager(similarity_threshold=0.80)

# Aquí implementar la captura de audio y procesamiento...
```

## API Reference

### Módulo `transcriber`

Contiene las funciones básicas para transcripción de voz en español.

#### Funciones Principales:
- `main()` - Función principal que inicia el proceso de transcripción.
- `callback(indata, frames, time, status)` - Callback para capturar audio.

### Módulo `transcriber_speaker`

Contiene las funciones para transcripción con identificación de hablantes.

#### Clases:
- `SpeakerManager` - Gestiona la identificación de hablantes.

#### Métodos de `SpeakerManager`:
- `extract_voice_features(audio_segment)` - Extrae características acústicas básicas.
- `extract_embedding(audio_segment)` - Extrae embeddings de audio.
- `identify_speaker(audio_segment)` - Identifica a qué hablante pertenece un segmento de audio.

#### Funciones Principales:
- `main()` - Función principal que inicia el proceso de transcripción con identificación.
- `cargar_modelos()` - Carga los modelos necesarios.
- `analyze_speaker_change(speaker_manager)` - Analiza cambios de hablante.
- `format_speaker_output(speaker, text)` - Formatea la salida con información del hablante.
