# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir a Pregunta a tus Finanzas! Este proyecto es open source y toda ayuda es bienvenida.

## 📋 Cómo Contribuir

### 1. Reportar Bugs
- Usa el [template de issues](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/issues/new)
- Incluye:
  - Descripción clara del problema
  - Pasos para reproducir
  - Comportamiento esperado vs actual
  - Logs si es posible

### 2. Sugerir Mejoras
- Abre una [discusión](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/discussions)
- Explica el caso de uso
- Propón una solución si la tienes

### 3. Añadir Soporte para Nuevos Bancos

Este es el tipo de contribución más valioso. Para añadir un nuevo banco:

#### Paso 1: Crear el Extractor

```python
# src/extractors/tu_banco_extractor.py
from src.extractors.base_extractor import BaseExtractor

class TuBancoExtractor(BaseExtractor):
    def extract_from_csv(self, file_path):
        # Implementar lógica específica del banco
        pass
```

#### Paso 2: Añadir Tests

```python
# tests/test_tu_banco.py
def test_tu_banco_extraction():
    extractor = TuBancoExtractor()
    result = extractor.extract_from_csv("sample.csv")
    assert len(result) > 0
```

#### Paso 3: Documentar

Añadir en `docs/SUPPORTED_BANKS.md`:
- Formato del CSV/Excel esperado
- Peculiaridades del banco
- Ejemplo de uso

### 4. Mejorar la Anonimización

Si encuentras PII no detectados:
1. Añade el patrón en `src/processors/adaptive_anonymizer.py`
2. Incluye tests para el nuevo patrón
3. Documenta el tipo de PII

## 🔧 Configuración del Entorno de Desarrollo

```bash
# 1. Fork y clonar
git clone https://github.com/tu-usuario/pregunta-a-tus-finanzas
cd pregunta-a-tus-finanzas

# 2. Crear rama
git checkout -b feature/nuevo-banco-santander

# 3. Instalar en modo desarrollo
pip install -r requirements.txt
pip install -e .

# 4. Ejecutar tests
pytest tests/
```

## 📝 Estándares de Código

- **Python 3.8+**
- **PEP 8** para estilo
- **Type hints** cuando sea posible
- **Docstrings** en todas las funciones públicas
- **Tests** para nueva funcionalidad

### Ejemplo de Docstring

```python
def extract_transactions(file_path: str) -> List[Dict]:
    """
    Extrae transacciones de un archivo bancario.
    
    Args:
        file_path: Ruta al archivo CSV/Excel
        
    Returns:
        Lista de diccionarios con las transacciones
        
    Raises:
        ValueError: Si el formato no es válido
    """
```

## 🚀 Proceso de Pull Request

1. **Fork** el repositorio
2. **Crea una rama** descriptiva: `feature/banco-santander`
3. **Commits atómicos** con mensajes claros
4. **Tests pasando** (incluye nuevos si es necesario)
5. **Actualiza docs** si cambias funcionalidad
6. **PR con descripción** completa

### Template de PR

```markdown
## Descripción
Breve descripción del cambio

## Tipo de cambio
- [ ] Bug fix
- [ ] Nueva funcionalidad
- [ ] Mejora de rendimiento
- [ ] Documentación

## Testing
- [ ] Tests existentes pasan
- [ ] Nuevos tests añadidos
- [ ] Probado manualmente

## Checklist
- [ ] Código sigue los estándares
- [ ] Documentación actualizada
- [ ] Sin datos sensibles
```

## 🏆 Reconocimiento

Todos los contribuidores serán listados en:
- README.md
- CONTRIBUTORS.md
- Releases notes

## 📞 Contacto

- Issues: [GitHub Issues](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/issues)
- Discusiones: [GitHub Discussions](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/discussions)
- Email: albert.gil@yourtechtribe.com

## 📜 Licencia

Al contribuir, aceptas que tu código será licenciado bajo MIT License.