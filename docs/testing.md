# Documentación de Tests Automatizados

## Descripción General

Este proyecto incluye una suite completa de tests automatizados para el sistema **hispano_transcriber**. Los tests están organizados en diferentes categorías para cubrir todos los aspectos del sistema de transcripción de voz con identificación de hablantes.

## Estructura de Tests

### 📁 tests/
```
tests/
├── __init__.py                 # Inicialización del paquete de tests
├── test_transcriber.py         # Tests para transcripción básica
├── test_transcriber_speaker.py # Tests para transcripción con hablantes
├── test_speaker_manager.py     # Tests para gestión de hablantes
├── test_configuration.py       # Tests de configuración y estructura
├── test_end_to_end.py         # Tests de integración completos
├── test_performance.py        # Tests de rendimiento y benchmarks
├── test_edge_cases.py         # Tests de casos extremos
├── run_all_tests.py           # Runner principal de tests
└── run_pytest.py              # Runner con pytest y cobertura
```

## Tipos de Tests

### 1. Tests Unitarios (`test_transcriber.py`, `test_speaker_manager.py`)
- **Propósito**: Verificar el funcionamiento de componentes individuales
- **Cobertura**: Funciones específicas, manejo de errores, casos básicos
- **Tiempo de ejecución**: Rápido (< 30 segundos)

### 2. Tests de Integración (`test_transcriber_speaker.py`, `test_end_to_end.py`)
- **Propósito**: Verificar la interacción entre componentes
- **Cobertura**: Flujo completo de transcripción, identificación de hablantes
- **Tiempo de ejecución**: Moderado (< 2 minutos)

### 3. Tests de Configuración (`test_configuration.py`)
- **Propósito**: Verificar estructura del proyecto y dependencias
- **Cobertura**: Archivos requeridos, imports, configuración
- **Tiempo de ejecución**: Rápido (< 15 segundos)

### 4. Tests de Rendimiento (`test_performance.py`)
- **Propósito**: Medir y verificar el rendimiento del sistema
- **Cobertura**: Latencia, throughput, uso de memoria
- **Tiempo de ejecución**: Variable (30 segundos - 5 minutos)

### 5. Tests de Casos Extremos (`test_edge_cases.py`)
- **Propósito**: Verificar robustez ante condiciones límite
- **Cobertura**: Audio corrupto, casos extremos, condiciones límite
- **Tiempo de ejecución**: Moderado (< 1 minuto)

## Cómo Ejecutar los Tests

### Opción 1: Script de PowerShell (Recomendado para Windows)
```powershell
# Todos los tests básicos
.\run_tests.ps1

# Tests con cobertura de código
.\run_tests.ps1 -Coverage

# Solo tests rápidos
.\run_tests.ps1 -Quick

# Incluir tests de rendimiento
.\run_tests.ps1 -Performance

# Ver ayuda
.\run_tests.ps1 -Help
```

### Opción 2: Script Batch (Windows)
```cmd
run_tests.bat
```

### Opción 3: Python con unittest
```bash
# Todos los tests
python -m unittest discover -s tests -p "test_*.py" -v

# Tests específicos
python -m unittest tests.test_transcriber -v
```

### Opción 4: Pytest (Recomendado para desarrollo)
```bash
# Instalar pytest
pip install pytest pytest-cov

# Todos los tests
pytest tests/ -v

# Tests con cobertura
pytest tests/ --cov=src/hispano_transcriber --cov-report=html

# Tests rápidos (excluir lentos)
pytest tests/ -m "not slow"

# Solo tests de rendimiento
pytest tests/test_performance.py -v
```

## Marcadores de Pytest

Los tests están organizados con marcadores para facilitar la ejecución selectiva:

- `@pytest.mark.slow`: Tests que tardan más tiempo
- `@pytest.mark.integration`: Tests de integración
- `@pytest.mark.unit`: Tests unitarios
- `@pytest.mark.speaker`: Tests relacionados con identificación de hablantes
- `@pytest.mark.transcription`: Tests relacionados con transcripción

### Ejemplos de uso:
```bash
# Solo tests unitarios rápidos
pytest -m "unit and not slow"

# Solo tests de hablantes
pytest -m "speaker"

# Excluir tests lentos
pytest -m "not slow"
```

## Configuración de Tests

### pytest.ini
Configuración principal de pytest con:
- Rutas de tests
- Patrones de archivos
- Opciones por defecto
- Marcadores definidos
- Filtros de warnings

### Dependencias de Testing
Definidas en `requirements-dev.txt`:
- `pytest>=7.0.0`: Framework de testing
- `pytest-cov>=2.12.0`: Cobertura de código
- `pytest-mock>=3.6.1`: Mocking avanzado

## Reportes de Cobertura

### Generar Reporte HTML
```bash
pytest tests/ --cov=src/hispano_transcriber --cov-report=html:htmlcov
```
El reporte se genera en `htmlcov/index.html`

### Generar Reporte en Terminal
```bash
pytest tests/ --cov=src/hispano_transcriber --cov-report=term-missing
```

### Generar Reporte XML (para CI/CD)
```bash
pytest tests/ --cov=src/hispano_transcriber --cov-report=xml
```

## Integración Continua (CI/CD)

### GitHub Actions
Configurado en `.github/workflows/tests.yml`:
- Tests en múltiples OS (Ubuntu, Windows, macOS)
- Tests en múltiples versiones de Python (3.8-3.11)
- Reporte de cobertura automático
- Tests de rendimiento en push a main

### Configuración Local para CI
```bash
# Simular entorno CI localmente
pytest tests/ --cov=src/hispano_transcriber --cov-report=xml -m "not slow"
```

## Mocking y Simulación

Los tests utilizan mocking extensivo para:
- **Aislamiento**: Tests unitarios sin dependencias externas
- **Velocidad**: Evitar operaciones costosas (carga de modelos)
- **Determinismo**: Resultados predecibles
- **Cobertura**: Simular condiciones específicas

### Componentes Mockeados:
- Modelos Vosk y Wav2Vec2
- Dispositivos de audio (sounddevice)
- Sistema de archivos
- Queues de audio

## Métricas y Benchmarks

### Tests de Rendimiento Incluyen:
1. **Latencia de transcripción**: Tiempo de procesamiento por segmento
2. **Throughput**: Capacidad de procesamiento en tiempo real
3. **Uso de memoria**: Estabilidad durante uso prolongado
4. **Procesamiento concurrente**: Rendimiento en multiples hilos
5. **Extracción de características**: Tiempo de análisis de audio

### Umbrales de Rendimiento:
- Latencia promedio: < 2 segundos por segmento
- Throughput: > 0.5x tiempo real
- Memoria: Sin crecimiento descontrolado
- Concurrencia: Speedup > 1x con múltiples hilos

## Casos de Prueba Específicos

### Audio Edge Cases:
- Audio vacío o muy corto
- Audio silencioso
- Audio extremadamente fuerte
- Audio con clipping
- Audio con valores NaN/infinitos
- Frecuencias extremas

### Condiciones del Sistema:
- Modelos no encontrados
- Errores de dispositivos de audio
- Condiciones de poca memoria
- Procesamiento concurrente
- Múltiples hablantes

### Validación de Datos:
- JSON malformado
- Embeddings corruptos
- Parámetros inválidos
- Configuraciones extremas

## Troubleshooting

### Problemas Comunes:

1. **Error: "Modelo no encontrado"**
   ```
   Solución: Descargar el modelo Vosk o usar mocks en tests
   ```

2. **Error: "sounddevice no disponible"**
   ```
   Solución: Los tests están diseñados para funcionar sin hardware de audio
   ```

3. **Tests lentos**
   ```
   Solución: Usar pytest -m "not slow" para excluir tests lentos
   ```

4. **Errores de memoria en tests largos**
   ```
   Solución: Ejecutar tests en lotes más pequeños
   ```

## Mejores Prácticas

### Para Desarrolladores:
1. **Ejecutar tests antes de commit**
2. **Mantener cobertura > 80%**
3. **Agregar tests para nuevas funcionalidades**
4. **Usar mocking apropiado**
5. **Documentar tests complejos**

### Para Tests de Rendimiento:
1. **Ejecutar en entorno consistente**
2. **Comparar con baselines anteriores**
3. **Considerar variabilidad del sistema**
4. **Documentar configuración de hardware**

## Extensión de Tests

### Agregar Nuevos Tests:
1. Crear archivo `test_nueva_funcionalidad.py`
2. Seguir convenciones de nomenclatura
3. Usar fixtures apropiados
4. Agregar marcadores relevantes
5. Actualizar documentación

### Ejemplo de Test Nuevo:
```python
import unittest
import pytest
from unittest.mock import Mock, patch

class TestNuevaFuncionalidad(unittest.TestCase):
    """Tests para nueva funcionalidad."""
    
    def setUp(self):
        """Preparación para tests."""
        pass
    
    @pytest.mark.unit
    def test_funcionalidad_basica(self):
        """Prueba funcionalidad básica."""
        # Implementar test
        pass
    
    @pytest.mark.slow
    def test_funcionalidad_compleja(self):
        """Prueba funcionalidad compleja."""
        # Implementar test que tarda más tiempo
        pass
```

---

## Contacto y Soporte

Para preguntas sobre los tests o problemas específicos:
1. Revisar esta documentación
2. Ejecutar tests con `-v` para más detalles
3. Consultar logs de CI/CD en GitHub Actions
4. Reportar issues específicos con ejemplos reproducibles
