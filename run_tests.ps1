# Script PowerShell para ejecutar tests automatizados
# Uso: .\run_tests.ps1 [-Coverage] [-Performance] [-Quick]

param(
    [switch]$Coverage,      # Generar reporte de cobertura
    [switch]$Performance,   # Ejecutar tests de rendimiento
    [switch]$Quick,         # Ejecutar solo tests rápidos
    [switch]$Help          # Mostrar ayuda
)

function Show-Help {
    Write-Host @"
HISPANO_TRANSCRIBER - Script de Tests Automatizados

USO:
    .\run_tests.ps1 [OPCIONES]

OPCIONES:
    -Coverage       Generar reporte de cobertura de código
    -Performance    Ejecutar tests de rendimiento (lentos)
    -Quick          Ejecutar solo tests rápidos
    -Help           Mostrar esta ayuda

EJEMPLOS:
    .\run_tests.ps1                    # Ejecutar todos los tests básicos
    .\run_tests.ps1 -Coverage          # Tests con cobertura
    .\run_tests.ps1 -Quick             # Solo tests rápidos
    .\run_tests.ps1 -Performance       # Incluir tests de rendimiento

"@
}

function Test-PythonAvailable {
    try {
        $version = python --version 2>&1
        Write-Host "✓ Python disponible: $version" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Error: Python no está instalado o no está en PATH" -ForegroundColor Red
        return $false
    }
}

function Install-Dependencies {
    Write-Host "`n📦 Instalando dependencias de desarrollo..." -ForegroundColor Cyan
    
    try {
        python -m pip install -q -r requirements-dev.txt
        Write-Host "✓ Dependencias instaladas correctamente" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠ Advertencia: No se pudieron instalar algunas dependencias" -ForegroundColor Yellow
    }
}

function Test-ProjectStructure {
    $requiredPaths = @(
        "src\hispano_transcriber",
        "tests",
        "requirements.txt",
        "requirements-dev.txt"
    )
    
    foreach ($path in $requiredPaths) {
        if (-not (Test-Path $path)) {
            Write-Host "✗ Error: No se encontró $path" -ForegroundColor Red
            Write-Host "Asegúrese de ejecutar este script desde la raíz del proyecto" -ForegroundColor Yellow
            return $false
        }
    }
    
    Write-Host "✓ Estructura del proyecto verificada" -ForegroundColor Green
    return $true
}

function Run-UnittestSuite {
    Write-Host "`n" + ("="*60) -ForegroundColor Cyan
    Write-Host "EJECUTANDO TESTS CON UNITTEST" -ForegroundColor Cyan
    Write-Host ("="*60) -ForegroundColor Cyan
    
    $testPattern = if ($Quick) { "test_transcriber.py test_configuration.py" } else { "test_*.py" }
    
    try {
        if ($Quick) {
            python -m unittest tests.test_transcriber tests.test_configuration -v
        } else {
            python -m unittest discover -s tests -p "test_*.py" -v
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✓ Tests unittest: EXITOSOS" -ForegroundColor Green
            return $true
        } else {
            Write-Host "`n✗ Tests unittest: FALLARON" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "`n✗ Error ejecutando unittest: $_" -ForegroundColor Red
        return $false
    }
}

function Run-PytestSuite {
    Write-Host "`n" + ("="*60) -ForegroundColor Cyan
    Write-Host "EJECUTANDO TESTS CON PYTEST" -ForegroundColor Cyan
    Write-Host ("="*60) -ForegroundColor Cyan
    
    # Verificar si pytest está disponible
    try {
        python -c "import pytest" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "pytest no disponible. Instalando..." -ForegroundColor Yellow
            python -m pip install pytest pytest-cov pytest-mock
        }
    }
    catch {
        Write-Host "Instalando pytest..." -ForegroundColor Yellow
        python -m pip install pytest pytest-cov pytest-mock
    }
    
    # Construir comando pytest
    $pytestArgs = @("tests/", "-v", "--tb=short")
    
    if ($Coverage) {
        $pytestArgs += @("--cov=src/hispano_transcriber", "--cov-report=html:htmlcov", "--cov-report=term-missing")
    }
    
    if ($Quick) {
        $pytestArgs += @("-m", "not slow")
    }
    
    if ($Performance) {
        # No agregar filtros, ejecutar todos incluyendo lentos
    } else {
        # Excluir tests de rendimiento por defecto
        $pytestArgs += @("--ignore=tests/test_performance.py")
    }
    
    try {
        $command = "python -m pytest " + ($pytestArgs -join " ")
        Write-Host "Ejecutando: $command" -ForegroundColor Gray
        
        Invoke-Expression $command
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✓ Tests pytest: EXITOSOS" -ForegroundColor Green
            if ($Coverage) {
                Write-Host "📊 Reporte de cobertura generado en: htmlcov\index.html" -ForegroundColor Cyan
            }
            return $true
        } else {
            Write-Host "`n✗ Tests pytest: FALLARON" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "`n✗ Error ejecutando pytest: $_" -ForegroundColor Red
        return $false
    }
}

function Run-PerformanceTests {
    if (-not $Performance) {
        return $true
    }
    
    Write-Host "`n" + ("="*60) -ForegroundColor Cyan
    Write-Host "TESTS DE RENDIMIENTO" -ForegroundColor Cyan
    Write-Host ("="*60) -ForegroundColor Cyan
    
    $response = Read-Host "Los tests de rendimiento pueden tardar varios minutos. ¿Continuar? (s/n)"
    if ($response -ne "s" -and $response -ne "S") {
        Write-Host "Tests de rendimiento omitidos." -ForegroundColor Yellow
        return $true
    }
    
    try {
        # Ejecutar tests de rendimiento rápidos
        python -m pytest tests/test_performance.py -v -m "not slow" --tb=short
        $quickPerfResult = $LASTEXITCODE
        
        # Preguntar por tests lentos
        $response = Read-Host "`n¿Ejecutar también tests lentos de rendimiento? (s/n)"
        if ($response -eq "s" -or $response -eq "S") {
            python -m pytest tests/test_performance.py -v -m "slow" --tb=short
        }
        
        return $quickPerfResult -eq 0
    }
    catch {
        Write-Host "Error en tests de rendimiento: $_" -ForegroundColor Red
        return $false
    }
}

function Show-Summary {
    param(
        [bool]$UnittestResult,
        [bool]$PytestResult,
        [bool]$PerformanceResult
    )
    
    Write-Host "`n" + ("="*60) -ForegroundColor Cyan
    Write-Host "RESUMEN DE RESULTADOS" -ForegroundColor Cyan
    Write-Host ("="*60) -ForegroundColor Cyan
    
    if ($UnittestResult) {
        Write-Host "✓ Tests unittest: EXITOSOS" -ForegroundColor Green
    } else {
        Write-Host "✗ Tests unittest: FALLARON" -ForegroundColor Red
    }
    
    if ($PytestResult) {
        Write-Host "✓ Tests pytest: EXITOSOS" -ForegroundColor Green
    } else {
        Write-Host "✗ Tests pytest: FALLARON" -ForegroundColor Red
    }
    
    if ($Performance) {
        if ($PerformanceResult) {
            Write-Host "✓ Tests rendimiento: EXITOSOS" -ForegroundColor Green
        } else {
            Write-Host "✗ Tests rendimiento: FALLARON" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    if ($UnittestResult -and $PytestResult) {
        Write-Host "🎉 TODOS LOS TESTS PRINCIPALES PASARON" -ForegroundColor Green
        return 0
    } else {
        Write-Host "❌ ALGUNOS TESTS FALLARON" -ForegroundColor Red
        return 1
    }
}

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

function Main {
    if ($Help) {
        Show-Help
        return 0
    }
    
    Write-Host @"
=========================================================
HISPANO_TRANSCRIBER - TESTS AUTOMATIZADOS
=========================================================
"@ -ForegroundColor Cyan
    
    Write-Host "Directorio actual: $(Get-Location)" -ForegroundColor Gray
    
    # Verificaciones previas
    if (-not (Test-PythonAvailable)) { return 1 }
    if (-not (Test-ProjectStructure)) { return 1 }
    
    # Instalar dependencias
    Install-Dependencies
    
    # Ejecutar tests
    $unittestResult = Run-UnittestSuite
    $pytestResult = Run-PytestSuite
    $performanceResult = Run-PerformanceTests
    
    # Mostrar resumen
    return Show-Summary $unittestResult $pytestResult $performanceResult
}

# Ejecutar función principal
$exitCode = Main
Write-Host "`nPresiona Enter para continuar..." -ForegroundColor Gray
Read-Host
exit $exitCode
