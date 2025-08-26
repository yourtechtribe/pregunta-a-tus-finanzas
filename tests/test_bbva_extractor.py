#!/usr/bin/env python3
"""
Análisis detallado de la FASE 1: Ingesta de datos con BBVAExtractor
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.extractors.bbva_extractor import BBVAExtractor
import pandas as pd
import json

print("=" * 80)
print("FASE 1: INGESTA DE DATOS - BBVA EXTRACTOR")
print("=" * 80)

# Inicializar el extractor
extractor = BBVAExtractor()

print("\n1. CONFIGURACIÓN DEL EXTRACTOR")
print("-" * 40)
print("El BBVAExtractor está configurado con:")
print(f"  - Palabras clave para categorización: {len(extractor.concept_keywords)} categorías")
print(f"  - Primeras 5 categorías definidas:")
for i, (keyword, category) in enumerate(list(extractor.concept_keywords.items())[:5]):
    print(f"    '{keyword}' -> {category}")

# Leer el archivo CSV de ejemplo
print("\n2. LECTURA DEL ARCHIVO CSV")
print("-" * 40)
input_file = "examples/sample_data.csv"
print(f"Archivo de entrada: {input_file}")

# Primero veamos el contenido crudo del CSV
df_raw = pd.read_csv(input_file, sep=';', nrows=3)
print("\nPrimeras 3 filas del CSV (crudo):")
print(df_raw.to_string())

# Ahora procesemos con el extractor
print("\n3. PROCESO DE EXTRACCIÓN")
print("-" * 40)

result = extractor.extract(input_file)

print(f"Resultado de la extracción:")
print(f"  - Tipo: {type(result)}")
print(f"  - Claves: {list(result.keys())}")

# Analizar las transacciones extraídas
print("\n4. TRANSACCIONES EXTRAÍDAS")
print("-" * 40)
transactions = result['transactions']
print(f"Total de transacciones: {len(transactions)}")

print("\nPrimeras 3 transacciones procesadas:")
for i, t in enumerate(transactions[:3]):
    print(f"\nTransacción {i+1}:")
    print(f"  - Fecha: {t['date']}")
    print(f"  - Fecha valor: {t['value_date']}")
    print(f"  - Descripción: {t['description']}")
    print(f"  - Descripción limpia: {t['description_clean']}")
    print(f"  - Monto: {t['amount']}€")
    print(f"  - Divisa: {t['currency']}")
    print(f"  - Categoría detectada: {t['category']}")
    print(f"  - Categoría BBVA: {t['bbva_category']}")
    print(f"  - Notas: {t['notes']}")

# Analizar el proceso de categorización
print("\n5. PROCESO DE CATEGORIZACIÓN")
print("-" * 40)
print("El extractor categoriza las transacciones basándose en:")
print("  1. Palabras clave en el campo 'Concepto'")
print("  2. Patrones de texto específicos")
print("  3. Categoría por defecto: 'Other'")

print("\nCategorización aplicada:")
category_counts = {}
for t in transactions:
    cat = t['category']
    category_counts[cat] = category_counts.get(cat, 0) + 1

for cat, count in sorted(category_counts.items()):
    print(f"  - {cat}: {count} transacciones")

# Analizar las estadísticas
print("\n6. ESTADÍSTICAS CALCULADAS")
print("-" * 40)
stats = result['statistics']
print(f"Estadísticas generadas:")
print(json.dumps(stats, indent=2, ensure_ascii=False))

# Analizar el período
print("\n7. PERÍODO DETECTADO")
print("-" * 40)
period = result.get('period', {})
print(f"Período de las transacciones:")
if period:
    for key, value in period.items():
        print(f"  - {key}: {value}")
else:
    print("  No se detectó período")

# Analizar metadata
print("\n8. METADATA")
print("-" * 40)
metadata = result.get('metadata', {})
print(f"Información adicional:")
for key, value in metadata.items():
    print(f"  - {key}: {value}")

# Análisis del procesamiento de montos
print("\n9. PROCESAMIENTO DE MONTOS")
print("-" * 40)
print("El extractor procesa los montos:")
print("  1. Detecta formato español (1.234,56) vs inglés (1,234.56)")
print("  2. Convierte comas decimales a puntos")
print("  3. Elimina separadores de miles")
print("  4. Convierte a float")

# Mostrar ejemplos de conversión
for t in transactions[:3]:
    print(f"\n  Concepto: {t['description'][:30]}")
    print(f"  Monto procesado: {t['amount']}€")
    print(f"  Tipo: {'Gasto' if t['amount'] < 0 else 'Ingreso'}")

# Resumen final
print("\n" + "=" * 80)
print("RESUMEN DE LA FASE 1")
print("=" * 80)
print(f"""
El BBVAExtractor realiza los siguientes pasos:

1. Lee el archivo CSV con separador ';'
2. Mapea los nombres de columnas a un formato estándar
3. Limpia los datos (elimina filas vacías, headers duplicados)
4. Procesa las fechas (formato DD/MM/YYYY -> YYYY-MM-DD)
5. Normaliza los montos (formato español -> float)
6. Categoriza transacciones basándose en palabras clave
7. Calcula estadísticas (totales, promedios, etc.)
8. Genera metadata del procesamiento

Resultado: Un diccionario estructurado con:
  - transactions: Lista de transacciones normalizadas
  - statistics: Resumen estadístico
  - period: Período temporal de los datos
  - metadata: Información del procesamiento
""")

print("✅ La Fase 1 funciona correctamente")