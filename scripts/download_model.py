#!/usr/bin/env python3
"""
Script para descargar y configurar el modelo Vosk en español.
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm
import shutil

# URL del modelo Vosk en español
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip"
MODEL_DIR = "vosk-model-es-0.42"
ZIP_FILE = "vosk-model-es-0.42.zip"

def download_file(url, filename):
    """Descarga un archivo mostrando una barra de progreso."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filtrar keep-alive chunks
                    f.write(chunk)
                    progress.update(len(chunk))
    return filename

def main():
    """Función principal para descargar y configurar el modelo."""
    print("📥 Descargando modelo Vosk en español...")
    
    # Comprobar si el directorio del modelo ya existe
    if os.path.exists(MODEL_DIR):
        print(f"✅ Modelo ya existente en '{MODEL_DIR}'")
        return
    
    # Comprobar si ya tenemos el archivo ZIP
    if not os.path.exists(ZIP_FILE):
        try:
            download_file(MODEL_URL, ZIP_FILE)
        except Exception as e:
            print(f"❌ Error descargando modelo: {e}")
            return
    
    # Descomprimir el archivo
    print("🔄 Descomprimiendo modelo...")
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"✅ Modelo descomprimido en '{MODEL_DIR}'")
    except Exception as e:
        print(f"❌ Error descomprimiendo modelo: {e}")
        return
    
    print("✅ Configuración del modelo completada.")

if __name__ == "__main__":
    main()
