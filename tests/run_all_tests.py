"""
Test runner principal para ejecutar todos los tests del proyecto.
"""

import unittest
import sys
import os

# Agregar el directorio src al path para poder importar los módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def create_test_suite():
    """Crea una suite de tests que incluye todos los tests del proyecto."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
      # Cargar todos los tests
    test_modules = [
        'test_transcriber',
        'test_transcriber_speaker', 
        'test_speaker_manager',
        'test_configuration',
        'test_end_to_end',
        'test_performance',
        'test_edge_cases'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Warning: No se pudo cargar el módulo de tests {module_name}: {e}")
    
    return suite


def run_tests(verbosity=2):
    """Ejecuta todos los tests con el nivel de verbosidad especificado."""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Retornar True si todos los tests pasaron
    return result.wasSuccessful()


def main():
    """Función principal del test runner."""
    print("=" * 70)
    print("EJECUTANDO TESTS AUTOMATIZADOS PARA HISPANO_TRANSCRIBER")
    print("=" * 70)
    
    # Verificar que estamos en el directorio correcto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    print(f"Directorio de tests: {current_dir}")
    print(f"Directorio del proyecto: {project_root}")
    print("-" * 70)
    
    # Ejecutar tests
    success = run_tests(verbosity=2)
    
    print("-" * 70)
    if success:
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        return 0
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        return 1


if __name__ == '__main__':
    sys.exit(main())
