---
name: Nuevo Parser Bancario
about: Contribuir con un parser para un nuevo banco
title: '[PARSER] Añadir soporte para [NOMBRE_BANCO]'
labels: 'enhancement, bank-parser'
assignees: ''

---

## Banco a Añadir
**Nombre del Banco**: [ej: Santander, CaixaBank, ING, etc.]
**País**: [España, México, etc.]

## Formato del Extracto
- **Formato de archivo**: [ ] CSV  [ ] Excel  [ ] PDF
- **Encoding**: [ej: UTF-8, ISO-8859-1]
- **Separador** (si CSV): [ej: coma, punto y coma, tabulador]
- **Formato decimal**: [ej: punto (1234.56) o coma (1234,56)]

## Estructura de Columnas
<!-- Lista las columnas del extracto bancario -->
| Columna | Nombre en Archivo | Descripción | Ejemplo |
|---------|------------------|-------------|---------|
| Fecha | | | DD/MM/YYYY |
| Concepto | | | |
| Importe | | | |
| Saldo | | | |

## Ejemplo del Extracto
```csv
Pega aquí las primeras 5 líneas del CSV/Excel (con datos anonimizados)
```

## Peculiaridades del Formato
<!-- ¿Hay algo especial en este formato que debamos saber? -->
- [ ] Tiene filas de cabecera que hay que saltar
- [ ] Incluye filas de resumen al final
- [ ] Formato de fecha no estándar
- [ ] Otros: 

## Implementación Propuesta
<!-- Si ya tienes código, compártelo aquí -->
```python
# Tu código del parser aquí (opcional)
```

## Archivos de Prueba
<!-- IMPORTANTE: Asegúrate de anonimizar TODOS los datos personales -->
- [ ] Puedo proporcionar un extracto de ejemplo anonimizado
- [ ] Incluiré tests unitarios

## Checklist de Contribución
- [ ] He leído la guía de contribución
- [ ] El parser maneja correctamente los formatos numéricos
- [ ] El parser incluye manejo de errores
- [ ] He probado con al menos 3 meses de datos
- [ ] Los datos de prueba están completamente anonimizados

## Información Adicional
<!-- ¿Algo más que debamos saber sobre este banco o formato? -->