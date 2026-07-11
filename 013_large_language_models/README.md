# Unidad 013 — Large Language Models

**Tecnicatura Superior en Ciencias de Datos e IA — IFTS24**  
Laboratorio de Procesamiento del Lenguaje Natural · Matías Barreto, 2026

Colección de notebooks sobre modelos de lenguaje grandes (LLMs), tokenización, embeddings, arquitectura Transformer, técnicas de decodificación y generación de texto avanzadas, y la implementación paso a paso de un modelo GPT desde cero utilizando PyTorch.

---

## Contenido

| Notebook | Tema |
|---|---|
| `01_introduccion_modelos_lenguaje` | Introducción a los Modelos de Lenguaje y conceptos fundamentales |
| `02_tokens_y_embeddings` | Procesamiento de texto: tokenización, subwords y vectores de embeddings |
| `04_aplicaciones_transformers` | Tareas clásicas con pipelines de Transformers (clasificación, NER, traducción, etc.) |

---

## Cómo descargar esta carpeta

Esta es una subcarpeta dentro de un repositorio más grande. Para descargarla sola, sin clonar todo el repositorio, hay dos herramientas web gratuitas que no requieren registro:

### Opción 1 — Download Directory

Una de las opciones más limpias y directas:

1. Copiá el enlace completo de esta carpeta en GitHub.
2. Entrá en **[download-directory.github.io](https://download-directory.github.io)**.
3. Pegá la URL en el cuadro de búsqueda y presioná Enter.
4. Se va a descargar automáticamente un archivo `.zip` con únicamente esta carpeta y todos sus archivos.

### Opción 2 — DownGit

Alternativa clásica y muy confiable:

1. Entrá en **[downgit.github.io](https://downgit.github.io)**.
2. Pegá el enlace de la carpeta en el campo que dice *GitHub URL*.
3. Hacé clic en el botón **Download**.

---

## Configuración del entorno local

### Requisitos previos

- **Python 3.10 o superior** — [python.org/downloads](https://www.python.org/downloads/)
- **uv** — gestor de entornos virtuales ultrarrápido

### Pasos para configurar el entorno

1. **Crear el entorno virtual** e instalar las dependencias con `uv`:
   ```bash
   # Crear entorno virtual .venv
   uv venv

   # Activar el entorno virtual
   # En macOS/Linux:
   source .venv/bin/activate
   # En Windows:
   .venv\Scripts\activate

   # Instalar las dependencias específicas de esta unidad
   uv pip install -r requirements.txt
   ```

2. **Modelos de spaCy**:
   Si ejecutás el notebook `04_aplicaciones_transformers.ipynb`, recordá que requiere el modelo en español de spaCy:
   ```bash
   python -m spacy download es_core_news_sm
   ```
