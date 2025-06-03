@echo off
REM Batch script para ejecutar todos los tests en Windows

echo =========================================================
echo EJECUTANDO TESTS AUTOMATICOS PARA HISPANO_TRANSCRIBER
echo =========================================================
echo.

REM Verificar que estamos en el directorio correcto
if not exist "src\hispano_transcriber" (
    echo Error: No se encontro el directorio src\hispano_transcriber
    echo Asegurese de ejecutar este script desde la raiz del proyecto
    pause
    exit /b 1
)

REM Verificar que Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python no esta instalado o no esta en PATH
    pause
    exit /b 1
)

echo Directorio actual: %CD%
echo.

REM Instalar dependencias de desarrollo si no existen
echo Instalando dependencias de desarrollo...
python -m pip install -q -r requirements-dev.txt
if %errorlevel% neq 0 (
    echo Advertencia: No se pudieron instalar algunas dependencias de desarrollo
)
echo.

REM Opcion 1: Ejecutar con unittest
echo =========================================================
echo EJECUTANDO TESTS CON UNITTEST
echo =========================================================
echo.
python -m unittest discover -s tests -p "test_*.py" -v
set unittest_result=%errorlevel%
echo.

REM Opcion 2: Ejecutar con pytest si esta disponible
echo =========================================================
echo VERIFICANDO PYTEST
echo =========================================================
python -c "import pytest" >nul 2>&1
if %errorlevel% equ 0 (
    echo pytest disponible. Ejecutando tests con pytest...
    echo.
    python -m pytest tests/ -v --tb=short
    set pytest_result=%errorlevel%
) else (
    echo pytest no disponible. Instalando...
    python -m pip install pytest pytest-cov
    if %errorlevel% equ 0 (
        echo Ejecutando tests con pytest...
        python -m pytest tests/ -v --tb=short
        set pytest_result=%errorlevel%
    ) else (
        echo No se pudo instalar pytest
        set pytest_result=1
    )
)
echo.

REM Ejecutar tests de rendimiento por separado (opcionales)
echo =========================================================
echo TESTS DE RENDIMIENTO (OPCIONALES)
echo =========================================================
echo Estos tests pueden tardar mas tiempo...
set /p run_performance="Ejecutar tests de rendimiento? (s/n): "
if /i "%run_performance%"=="s" (
    echo Ejecutando tests de rendimiento...
    python -m pytest tests/test_performance.py -v -m "not slow" --tb=short
    echo.
    
    set /p run_slow="Ejecutar tests lentos de rendimiento? (s/n): "
    if /i "%run_slow%"=="s" (
        echo Ejecutando tests lentos...
        python -m pytest tests/test_performance.py -v -m "slow" --tb=short
    )
)
echo.

REM Mostrar resumen
echo =========================================================
echo RESUMEN DE RESULTADOS
echo =========================================================
if %unittest_result% equ 0 (
    echo ✓ Tests unittest: EXITOSOS
) else (
    echo ✗ Tests unittest: FALLARON
)

if defined pytest_result (
    if %pytest_result% equ 0 (
        echo ✓ Tests pytest: EXITOSOS
    ) else (
        echo ✗ Tests pytest: FALLARON
    )
)

echo.
if %unittest_result% equ 0 (
    echo ✅ TODOS LOS TESTS PRINCIPALES PASARON
    set final_result=0
) else (
    echo ❌ ALGUNOS TESTS FALLARON
    set final_result=1
)

echo.
echo Presione cualquier tecla para continuar...
pause >nul

exit /b %final_result%
