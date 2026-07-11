"""
Funciones auxiliares para el material de Salida Estructurada y RAG.

Este módulo provee la infraestructura para generación de texto,
parseo de JSON, obtención de documentos, fragmentación de corpus
y recuperación semántica.

Backend de inferencia: Ollama (modelo local, sin necesidad de GPU dedicada).
"""

import json
import time
import requests
import numpy as np
import ollama
from bs4 import BeautifulSoup
from typing import List, Dict
from IPython.display import HTML, display


# ---------------------------------------------------------------------------
# Tablas de resultados (visualización)
# ---------------------------------------------------------------------------

def make_results_table(resultados_por_issue, issues, n_test=10):
    """
    Genera una tabla HTML con los resultados de la Estrategia 1 (restricciones blandas).

    Muestra, para cada issue, cuántos de los 3 intentos pasaron
    el parseo estricto, el parseo leniente y la validación de esquema.
    """
    filas = ""
    for i, (issue, intentos) in enumerate(zip(issues[:n_test], resultados_por_issue)):
        titulo = issue['title'][:45]

        # Contamos cuántos intentos superaron cada criterio
        estricto = sum(r["strict_json"] for r in intentos)
        leniente = sum(r["lenient_json"] for r in intentos)
        esquema = sum(r["schema_ok"] for r in intentos)

        def barra(n, total=3):
            # Verde si todos pasan, naranja si algunos, rojo si ninguno
            color = "#2ecc71" if n == total else "#e67e22" if n > 0 else "#e74c3c"
            return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:4px;font-weight:bold">{n}/{total}</span>'

        filas += f"""
        <tr style="border-bottom:1px solid #eee">
            <td style="padding:6px 12px;color:#888">{i}</td>
            <td style="padding:6px 12px">{titulo}</td>
            <td style="padding:6px 12px;text-align:center">{barra(estricto)}</td>
            <td style="padding:6px 12px;text-align:center">{barra(leniente)}</td>
            <td style="padding:6px 12px;text-align:center">{barra(esquema)}</td>
        </tr>"""

    return HTML(f"""
    <h4 style="font-family:sans-serif">Estrategia 1: Restricciones Blandas — Resultados (temp=1.0, 3 intentos, parseo estricto)</h4>
    <table style="border-collapse:collapse;font-family:monospace;font-size:13px;width:100%">
      <tr style="background:#2c3e50;color:white">
        <th style="padding:8px 12px">#</th>
        <th style="padding:8px 12px;text-align:left">Issue</th>
        <th style="padding:8px 12px">Estricto</th>
        <th style="padding:8px 12px">Leniente</th>
        <th style="padding:8px 12px">Esquema</th>
      </tr>
      {filas}
    </table>
    <p style="font-family:sans-serif;font-size:12px;color:#555;margin-top:8px">
      <span style="background:#2ecc71;color:white;padding:1px 6px;border-radius:3px">3/3</span> siempre pasa &nbsp;
      <span style="background:#e67e22;color:white;padding:1px 6px;border-radius:3px">1–2/3</span> inconsistente &nbsp;
      <span style="background:#e74c3c;color:white;padding:1px 6px;border-radius:3px">0/3</span> siempre falla
    </p>
    """)


def mostrar_fallo(indice_issue, indice_intento, etiquetas, texto_crudo):
    """
    Muestra el detalle de un intento fallido: índice, etiquetas de error y
    el texto generado por el modelo (formateado si es JSON válido).
    """
    print(f"Issue {indice_issue} — Intento {indice_intento} ({', '.join(etiquetas)}):")

    # Limpiamos los delimitadores de markdown por si el modelo los agregó
    limpio = texto_crudo.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        print(json.dumps(json.loads(limpio), indent=2))
    except Exception:
        # Si no es JSON válido, mostramos los primeros 300 caracteres
        print(limpio[:300])

    print("-" * 50)


def make_cd_results_table(resultados_cd, issues, n_test=10, nombre_estrategia="Estrategia 2: Decodificación Restringida"):
    """
    Genera una tabla HTML con los resultados de la Estrategia 2 (decodificación restringida).

    Muestra para cada issue: si pasó, el tiempo de generación,
    el tipo de issue y la severidad extraídos.
    """
    filas = ""
    for i, (issue, resultado) in enumerate(zip(issues[:n_test], resultados_cd)):
        titulo = issue['title'][:45]
        exito = resultado["success"]
        tiempo = f"{resultado['time']:.1f}s"
        tipo_issue = resultado.get("issue_type", "—")
        severidad = resultado.get("severity", "—")

        # Indicador visual de éxito o fallo
        estado = (
            '<span style="background:#2ecc71;color:white;padding:2px 10px;border-radius:4px;font-weight:bold">✅ ok</span>'
            if exito else
            '<span style="background:#e74c3c;color:white;padding:2px 10px;border-radius:4px;font-weight:bold">❌ fallo</span>'
        )

        filas += f"""
        <tr style="border-bottom:1px solid #eee">
            <td style="padding:6px 12px;color:#888">{i}</td>
            <td style="padding:6px 12px">{titulo}</td>
            <td style="padding:6px 12px;text-align:center">{estado}</td>
            <td style="padding:6px 12px;text-align:center;color:#555">{tipo_issue}</td>
            <td style="padding:6px 12px;text-align:center;color:#555">{severidad}</td>
            <td style="padding:6px 12px;text-align:center;color:#888">{tiempo}</td>
        </tr>"""

    tasa_exito = sum(r["success"] for r in resultados_cd) / len(resultados_cd)
    color = "#2ecc71" if tasa_exito == 1.0 else "#e67e22" if tasa_exito > 0.5 else "#e74c3c"

    return HTML(f"""
    <h4 style="font-family:sans-serif">{nombre_estrategia} — Resultados</h4>
    <table style="border-collapse:collapse;font-family:monospace;font-size:13px;width:100%">
      <tr style="background:#2c3e50;color:white">
        <th style="padding:8px 12px">#</th>
        <th style="padding:8px 12px;text-align:left">Issue</th>
        <th style="padding:8px 12px">Estado</th>
        <th style="padding:8px 12px">Tipo</th>
        <th style="padding:8px 12px">Severidad</th>
        <th style="padding:8px 12px">Tiempo</th>
      </tr>
      {filas}
    </table>
    <p style="font-family:sans-serif;font-size:13px;margin-top:8px">
      Tasa de éxito: <span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{tasa_exito:.0%}</span>
    </p>
    """)


# ---------------------------------------------------------------------------
# Generación de texto (via Ollama)
# ---------------------------------------------------------------------------

def generate(mensajes, model_name, max_tokens=512, temperature=0.0):
    """
    Genera una respuesta del modelo usando Ollama.

    Acepta dos formatos de entrada:
    - Lista de mensajes estilo chat: [{"role": "user", "content": "..."}]
    - String simple: se envuelve automáticamente como mensaje de usuario.

    Parámetros:
        mensajes: str o lista de dicts con 'role' y 'content'
        model_name: nombre del modelo en Ollama (ej: "gemma4:e2b")
        max_tokens: cantidad máxima de tokens a generar
        temperature: 0.0 para decodificación greedy (determinística),
                     valores mayores activan el muestreo (más variedad)

    Retorna:
        str: el texto generado
    """
    if isinstance(mensajes, str):
        mensajes = [{"role": "user", "content": mensajes}]

    # Configuramos las opciones de generación
    opciones = {"num_predict": max_tokens, "temperature": temperature}

    respuesta = ollama.chat(
        model=model_name,
        messages=mensajes,
        options=opciones
    )
    return respuesta["message"]["content"]


def generate_structured(mensajes, model_name, schema_dict):
    """
    Genera una respuesta estructurada usando Ollama con un esquema JSON.

    Ollama interviene en la generación para garantizar que la salida
    cumpla con el esquema JSON provisto. Es el equivalente a la
    decodificación restringida de la librería `outlines`.

    Parámetros:
        mensajes: str o lista de dicts con 'role' y 'content'
        model_name: nombre del modelo en Ollama
        schema_dict: esquema JSON como dict (usar ParsedIssue.model_json_schema())

    Retorna:
        str: el texto JSON generado (garantizado válido según el esquema)
    """
    if isinstance(mensajes, str):
        mensajes = [{"role": "user", "content": mensajes}]

    respuesta = ollama.chat(
        model=model_name,
        messages=mensajes,
        format=schema_dict
    )
    return respuesta["message"]["content"]


# ---------------------------------------------------------------------------
# Parseo de JSON
# ---------------------------------------------------------------------------

def try_parse_strict(texto_crudo: str) -> dict | None:
    """
    Parseo JSON estricto: intenta parsear la salida del modelo sin ninguna limpieza previa.

    Si el modelo agregó delimitadores de markdown (```json ... ```) o
    texto adicional, este parseo fallará. Así funcionaría un sistema en producción
    que recibe el texto directamente y lo pasa a otro componente.

    Retorna el dict si el JSON es válido, None si no lo es.
    """
    try:
        return json.loads(texto_crudo.strip())
    except json.JSONDecodeError:
        return None


def try_parse_lenient(texto_crudo: str) -> dict | None:
    """
    Parseo JSON leniente: elimina delimitadores de markdown comunes antes de parsear.

    Más tolerante que el parseo estricto, pero sigue fallando si el JSON
    en sí está mal formado (llaves sin cerrar, comas de más, etc.).

    Retorna el dict si el JSON es válido, None si no lo es.
    """
    limpio = texto_crudo.strip()

    # Quitamos los delimitadores de código que los modelos suelen agregar
    for prefijo in ["```json", "```"]:
        if limpio.startswith(prefijo):
            limpio = limpio[len(prefijo):]

    if limpio.endswith("```"):
        limpio = limpio[:-3]

    try:
        return json.loads(limpio.strip())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Obtención de documentos (corpus)
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> Dict:
    """
    Descarga una página web y extrae su contenido textual principal.

    Usa requests para obtener el HTML y BeautifulSoup para parsearlo.
    Extrae el elemento <div role="main"> o <article> si existe,
    ya que suelen contener el contenido útil de la página.

    Parámetros:
        url: dirección de la página a descargar

    Retorna:
        dict con claves: "url" (str), "title" (str), "content" (str)
    """
    cabeceras = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; material-didactico/1.0)"
        )
    }

    respuesta = requests.get(url, headers=cabeceras, timeout=15)
    respuesta.raise_for_status()

    # BeautifulSoup parsea el HTML y nos permite navegar su estructura
    soup = BeautifulSoup(respuesta.content, "html.parser")

    # Buscamos el contenedor principal del contenido
    contenido = soup.find("div", {"role": "main"}) or soup.find("article")

    # Extraemos el título de la página
    titulo = soup.find("title").text.strip() if soup.find("title") else url

    # get_text extrae el texto plano, separando bloques con saltos de línea
    texto = contenido.get_text(separator="\n\n", strip=True) if contenido else ""

    # Limitamos a 15000 caracteres para no saturar el contexto del modelo
    return {"url": url, "title": titulo, "content": texto[:15000]}


# ---------------------------------------------------------------------------
# Fragmentación del corpus (chunking)
# ---------------------------------------------------------------------------

def chunk_corpus(corpus: List[Dict], chunk_fn, **kwargs) -> List[Dict]:
    """
    Fragmenta todos los documentos de un corpus usando la función indicada.

    Parámetros:
        corpus: lista de dicts con los documentos
        chunk_fn: función de fragmentación a aplicar a cada documento
        **kwargs: parámetros adicionales que se pasan a chunk_fn

    Retorna:
        lista de fragmentos (dicts), uno por cada trozo de cada documento
    """
    fragmentos = []

    for doc in corpus:
        # Extendemos la lista con los fragmentos de cada documento
        fragmentos.extend(chunk_fn(doc, **kwargs))

    return fragmentos


# ---------------------------------------------------------------------------
# Recuperador semántico (Retriever)
# ---------------------------------------------------------------------------

class Retriever:
    """
    Recuperador denso basado en similitud coseno sobre embeddings de oraciones.

    Al inicializarse, embebe todos los fragmentos del corpus.
    Al buscar, compara el embedding de la consulta con todos los fragmentos
    y devuelve los más similares.
    """

    def __init__(self, fragmentos: List[Dict], embedder):
        self.fragmentos = fragmentos
        self.embedder = embedder

        print(f"Embebiendo {len(fragmentos)} fragmentos...")

        # encode convierte cada fragmento de texto en un vector numérico
        self.embeddings = embedder.encode(
            [f["text"] for f in fragmentos], show_progress_bar=True
        )

        print("Recuperador listo.")

    def search(self, consulta: str, top_k: int = 5) -> List[Dict]:
        """
        Busca los fragmentos más relevantes para una consulta.

        Parámetros:
            consulta: pregunta o texto a buscar
            top_k: cuántos fragmentos devolver

        Retorna:
            lista de los top_k fragmentos más similares, con su puntaje
        """
        # Embebemos la consulta en el mismo espacio vectorial que los fragmentos
        embedding_consulta = self.embedder.encode([consulta])[0]

        # Calculamos la similitud coseno entre la consulta y todos los fragmentos
        # np.dot: producto punto; np.linalg.norm: norma (longitud) del vector
        similitudes = np.dot(self.embeddings, embedding_consulta) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(embedding_consulta) + 1e-10
        )

        # argsort ordena de menor a mayor; [::-1] invierte para mayor a menor
        indices_top = np.argsort(similitudes)[::-1][:top_k]

        return [{**self.fragmentos[i], "score": float(similitudes[i])} for i in indices_top]


# ---------------------------------------------------------------------------
# Embedder via Ollama
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """
    Adaptador para usar modelos de embeddings de Ollama con la clase Retriever.

    Ollama incluye modelos de embeddings dedicados (ej: mxbai-embed-large)
    que convierten texto en vectores numéricos para búsqueda semántica.
    Esta clase implementa el método .encode() para ser compatible
    con la clase Retriever existente.
    """

    def __init__(self, model_name="mxbai-embed-large:latest"):
        self.model_name = model_name
        print(f"Embedder listo: '{model_name}'")

    def encode(self, textos, show_progress_bar=False):
        """
        Convierte una lista de textos en vectores numéricos (embeddings).

        Parámetros:
            textos: lista de strings a embeber
            show_progress_bar: muestra progreso si es True

        Retorna:
            numpy array de shape (n_textos, dimensión_embedding)
        """
        embeddings = []
        total = len(textos)

        for i, texto in enumerate(textos):
            if show_progress_bar and (i % 10 == 0 or i == total - 1):
                print(f"  Embebiendo fragmento {i+1}/{total}...", end="\r")

            respuesta = ollama.embed(model=self.model_name, input=texto)
            embeddings.append(respuesta["embeddings"][0])

        if show_progress_bar:
            print(f"  {total}/{total} fragmentos embebidos. ✓")

        return np.array(embeddings)


# ---------------------------------------------------------------------------
# Previsualización del dataset
# ---------------------------------------------------------------------------

def display_issues_preview(issues, n=5):
    """
    Muestra una tabla HTML con una previsualización de los primeros n issues.

    Para cada issue muestra el índice, el título y un extracto del cuerpo.
    """
    filas = ""
    for i, issue in enumerate(issues[:n]):
        titulo = issue['title'][:70]

        # Colapsamos espacios múltiples y limitamos a 150 caracteres
        cuerpo = ' '.join(issue['body'].split())[:150]

        fondo = "#f9f9f9" if i % 2 == 0 else "white"

        filas += f"""
        <tr style="border-bottom:1px solid #eee;background:{fondo}">
            <td style="padding:8px 12px;color:#888;width:30px">{i}</td>
            <td style="padding:8px 12px;font-weight:bold;width:35%;vertical-align:top">{titulo}</td>
            <td style="padding:8px 12px;color:#555;vertical-align:top">{cuerpo}...</td>
        </tr>"""

    display(HTML(f"""
    <table style="border-collapse:collapse;font-family:sans-serif;font-size:12px;width:100%;table-layout:fixed">
      <tr style="background:#2c3e50;color:white">
        <th style="padding:8px 12px;width:30px">#</th>
        <th style="padding:8px 12px;text-align:left;width:35%">Título</th>
        <th style="padding:8px 12px;text-align:left">Extracto del cuerpo</th>
      </tr>
      {filas}
    </table>
    """))


# ---------------------------------------------------------------------------
# Ayudantes para visualización RAG
# ---------------------------------------------------------------------------

def display_chunks_summary(fragmentos, corpus):
    """
    Muestra un resumen del índice de fragmentos construido.

    Informa la cantidad de documentos, fragmentos, tamaños mínimo,
    máximo y promedio, y un aviso si algún fragmento supera los 1000 caracteres.
    También muestra el texto del primer fragmento como ejemplo.
    """
    import numpy as np

    tamaños = [len(f["text"]) for f in fragmentos]
    sobredimensionados = [s for s in tamaños if s > 1000]

    color_estado = "#e74c3c" if sobredimensionados else "#2ecc71"
    texto_estado = (
        f"⚠ {len(sobredimensionados)} fragmentos superan 1000 caracteres (mayor: {max(sobredimensionados)})"
        if sobredimensionados
        else "✓ Todos los fragmentos dentro del rango esperado"
    )

    display(HTML(f"""
    <div style="font-family:sans-serif;font-size:13px;padding:12px;background:#f8f9fa;border-radius:6px;border-left:4px solid #2c3e50">
      <b>Índice de fragmentos construido</b><br><br>
      📄 <b>{len(corpus)}</b> documentos → <b>{len(fragmentos)}</b> fragmentos<br>
      📏 Mín: <b>{min(tamaños)}</b> &nbsp;|&nbsp; Máx: <b>{max(tamaños)}</b> &nbsp;|&nbsp; Prom: <b>{np.mean(tamaños):.0f}</b> caracteres<br>
      <span style="color:{color_estado}"><b>{texto_estado}</b></span>
    </div>
    <br>
    <div style="font-family:monospace;font-size:12px;background:#fff;padding:10px;border:1px solid #eee;border-radius:4px">
      <b>Fragmento de muestra de '{fragmentos[0]['title'][:50]}':</b><br><br>
      {fragmentos[0]['text'][:300]}...
    </div>
    """))


def display_retrieval_results(consultas, retriever, top_k=3):
    """
    Muestra los resultados de recuperación para una lista de consultas.

    Para cada consulta, muestra una tabla con los top_k fragmentos
    recuperados, su puntaje de similitud, fuente y un extracto.
    """
    html = "<div style='font-family:sans-serif;font-size:13px'>"

    for consulta in consultas:
        resultados = retriever.search(consulta, top_k=top_k)
        filas = ""

        for i, r in enumerate(resultados, 1):
            # Coloreamos el puntaje según su nivel de similitud
            color_puntaje = "#2ecc71" if r['score'] > 0.6 else "#e67e22" if r['score'] > 0.5 else "#e57373"

            filas += f"""
            <tr style="border-bottom:1px solid #eee">
                <td style="padding:6px 10px;color:#888">{i}</td>
                <td style="padding:6px 10px;text-align:center">
                    <span style="background:{color_puntaje};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{r['score']:.3f}</span>
                </td>
                <td style="padding:6px 10px;color:#2c3e50">{r['title'][:45]}</td>
                <td style="padding:6px 10px;color:#555;font-family:monospace;font-size:11px">{r['text'][:100]}...</td>
            </tr>"""

        html += f"""
        <div style="margin-bottom:16px">
          <div style="background:#2c3e50;color:white;padding:8px 12px;border-radius:6px 6px 0 0;font-weight:bold">Consulta: {consulta}</div>
          <table style="border-collapse:collapse;width:100%;background:white;border:1px solid #eee;border-top:none;border-radius:0 0 6px 6px">
            <tr style="background:#f5f5f5;color:#888;font-size:11px">
              <th style="padding:6px 10px">#</th>
              <th style="padding:6px 10px">Puntaje</th>
              <th style="padding:6px 10px;text-align:left">Fuente</th>
              <th style="padding:6px 10px;text-align:left">Extracto</th>
            </tr>
            {filas}
          </table>
        </div>"""

    html += "</div>"
    display(HTML(html))