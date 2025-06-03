"""
Script para ejecutar tests con pytest y generar reporte de cobertura.
"""

import subprocess
import sys
import os


def run_pytest_with_coverage():
    """Ejecuta pytest con cobertura de código."""
    
    # Cambiar al directorio del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("EJECUTANDO TESTS CON PYTEST Y COBERTURA")
    print("=" * 70)
    print(f"Directorio de trabajo: {os.getcwd()}")
    print()
    
    # Comando pytest con cobertura
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--verbose",
        "--cov=src/hispano_transcriber",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--tb=short"
    ]
    
    try:
        # Ejecutar pytest
        result = subprocess.run(pytest_cmd, capture_output=False, text=True)
        
        print("\n" + "=" * 70)
        if result.returncode == 0:
            print("✅ TODOS LOS TESTS PASARON")
            print("📊 Reporte de cobertura generado en: htmlcov/index.html")
        else:
            print("❌ ALGUNOS TESTS FALLARON")
        
        return result.returncode
        
    except FileNotFoundError:
        print("❌ Error: pytest no está instalado.")
        print("Instala pytest con: pip install pytest pytest-cov")
        return 1
    except Exception as e:
        print(f"❌ Error ejecutando pytest: {e}")
        return 1


def install_test_dependencies():
    """Instala las dependencias necesarias para los tests."""
    dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=2.12.0",
        "pytest-mock>=3.6.1"
    ]
    
    print("Instalando dependencias de testing...")
    for dep in dependencies:
        cmd = [sys.executable, "-m", "pip", "install", dep]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {dep} instalado")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {dep}")


def main():
    """Función principal."""
    if len(sys.argv) > 1 and sys.argv[1] == "--install-deps":
        install_test_dependencies()
        return 0
    
    return run_pytest_with_coverage()


if __name__ == "__main__":
    sys.exit(main())
