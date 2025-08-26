# Contributing to Pregunta tus Finanzas

¡Gracias por tu interés en contribuir! Este proyecto es open source y todas las contribuciones son bienvenidas.

## 🚀 Quick Start para Contribuidores

1. **Fork el repositorio**
2. **Clone su fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pregunta-a-tus-finanzas
   cd pregunta-a-tus-finanzas
   ```

3. **Setup automático**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Configure su API key** en `.env`:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

5. **Ejecute los tests**:
   ```bash
   chmod +x run_tests.sh
   ./run_tests.sh
   ```

## 📝 Proceso de Contribución

1. **Cree una rama** para su feature:
   ```bash
   git checkout -b feature/mi-nueva-funcionalidad
   ```

2. **Haga sus cambios** siguiendo las guías de estilo

3. **Ejecute los tests** para verificar que todo funciona:
   ```bash
   ./run_tests.sh
   ```

4. **Commit sus cambios**:
   ```bash
   git commit -m "feat: Descripción de la nueva funcionalidad"
   ```

5. **Push a su fork**:
   ```bash
   git push origin feature/mi-nueva-funcionalidad
   ```

6. **Abra un Pull Request** desde GitHub

## 🎯 Áreas de Contribución

### Alta Prioridad
- 🏦 **Nuevos extractores bancarios**: Añadir soporte para más bancos
- 📄 **Extractor PDF**: Implementar extracción de PDFs
- 🧪 **Tests**: Aumentar cobertura de tests
- 📚 **Documentación**: Mejorar y traducir documentación

### Features Deseados
- 🌐 **Interfaz Web**: Dashboard con Streamlit/Gradio
- 📊 **Visualizaciones**: Gráficos y reportes mejorados
- 🤖 **Más modelos LLM**: Soporte para Claude, Llama, etc.
- 🔐 **Seguridad**: Mejoras en anonimización y encriptación

## 💻 Guías de Desarrollo

### Estructura del Código

```
src/
├── extractors/      # Parsers de extractos bancarios
├── processors/      # Procesadores de datos
├── agents/          # Agentes de categorización
└── rag/            # Implementación RAG/LightRAG
```

### Añadir un Nuevo Banco

1. Cree un nuevo archivo en `src/extractors/`:
   ```python
   # src/extractors/mi_banco_extractor.py
   class MiBancoExtractor(BaseExtractor):
       def extract(self, file_path):
           # Implementar lógica de extracción
           pass
   ```

2. Añada tests en `tests/`:
   ```python
   # tests/test_mi_banco_extractor.py
   def test_mi_banco_extraction():
       # Tests para el nuevo extractor
       pass
   ```

3. Documente el formato soportado en `docs/`

### Estilo de Código

- **Python**: Seguimos PEP 8
- **Docstrings**: Usar formato Google
- **Type hints**: Requeridos para funciones públicas
- **Tests**: Mínimo 80% cobertura para nuevas features

### Commit Messages

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bugs
- `docs:` Cambios en documentación
- `test:` Añadir o modificar tests
- `refactor:` Refactorización sin cambios funcionales
- `perf:` Mejoras de performance
- `chore:` Tareas de mantenimiento

## 🧪 Testing

### Ejecutar todos los tests:
```bash
./run_tests.sh
```

### Ejecutar tests específicos:
```bash
pytest tests/test_bbva_extractor.py -v
```

### Verificar cobertura:
```bash
pytest --cov=src tests/
```

## 📋 Checklist para Pull Requests

- [ ] El código sigue el estilo del proyecto
- [ ] He añadido tests para mi código
- [ ] Todos los tests pasan
- [ ] He actualizado la documentación
- [ ] He añadido una entrada en CHANGELOG.md
- [ ] El commit message sigue las convenciones

## 🐛 Reportar Bugs

Para reportar bugs, abra un [Issue](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/issues) con:

1. Descripción del problema
2. Pasos para reproducirlo
3. Comportamiento esperado vs actual
4. Logs o mensajes de error
5. Versión de Python y sistema operativo

## 💬 Preguntas y Soporte

- **Discusiones**: Use [GitHub Discussions](https://github.com/yourtechtribe/pregunta-a-tus-finanzas/discussions)
- **Chat**: Únase a nuestro [Discord/Slack](#) (próximamente)
- **Email**: albert.gil@yourtechtribe.com

## 📜 Licencia

Al contribuir, acepta que sus contribuciones serán licenciadas bajo la misma licencia MIT del proyecto.

## 🙏 Agradecimientos

¡Gracias a todos los contribuidores que hacen este proyecto posible!

---

**¿Listo para contribuir?** ¡Fork el proyecto y empiece hoy mismo! 🚀