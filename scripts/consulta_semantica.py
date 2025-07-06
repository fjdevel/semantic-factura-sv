# ============================
# Asistente Legal Semántico
# ============================
# Este script construye una interfaz de consulta legal para facturación electrónica
# usando modelos de lenguaje (LLM), embeddings semánticos y búsqueda vectorial.
#
# Tecnologías utilizadas:
# - SentenceTransformers: para convertir texto en vectores (embeddings).
# - PostgreSQL con pgvector: para almacenar y buscar documentos por similitud semántica.
# - Ollama + Qwen3: para generar respuestas a partir del contexto legal recuperado.
# - Gradio: para exponer todo como una interfaz web fácil de usar.

import os
import gradio as gr
import psycopg2
import numpy as np
import torch
import requests
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# =============================================
# 1. Configuración general y parámetros globales
# =============================================

# Datos de conexión a PostgreSQL (debe tener instalada la extensión pgvector)
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
    "dbname": "semantic_fiscal"
}

# Dimensión esperada del embedding generado (según el modelo usado)
EMBEDDING_DIM = 768

# Nombre del modelo de embeddings (preentrenado para español)
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-es"

# URL local del servidor Ollama que hospeda modelos LLM como Qwen3
OLLAMA_API = "http://localhost:11434/api/generate"

# =============================================
# 2. Cargar modelo de embeddings con SentenceTransformer
# =============================================
# Este modelo convierte cualquier texto (por ejemplo, una pregunta legal)
# en un vector numérico que representa su significado.

# Se detecta si hay GPU disponible, lo cual mejora significativamente el rendimiento
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargamos el modelo en el dispositivo adecuado
model = SentenceTransformer(EMBEDDING_MODEL, device=device)

# =============================================
# 3. Detectar modelo Qwen3 disponible en Ollama
# =============================================
# Ollama es un servidor local para ejecutar modelos LLM como Qwen3.
# Esta función consulta el endpoint /tags para listar los modelos disponibles
# y selecciona aquel cuyo nombre contenga "qwen3".

def obtener_modelo_qwen3():
    modelos = requests.get("http://localhost:11434/api/tags").json().get("models", [])
    for m in modelos:
        if "qwen3" in m["name"].lower():
            return m["name"]
    raise RuntimeError("Qwen3 no encontrado en Ollama")

# =============================================
# 4. Generar embeddings desde texto
# =============================================
# Esta función toma un fragmento de texto (por ejemplo, una pregunta)
# y genera su embedding. Un embedding es un vector de números que
# captura el significado del texto en un espacio semántico.

def generar_embedding(texto):
    emb = model.encode(texto, normalize_embeddings=True)
    return emb[:EMBEDDING_DIM]  # Se asegura que tenga la dimensión correcta

# =============================================
# 5. Buscar fragmentos similares en la base vectorial
# =============================================
# Esta función consulta la tabla "normativa" en PostgreSQL usando pgvector.
# Busca los k fragmentos más cercanos al embedding proporcionado,
# ordenados por similitud (distancia L2).

def consultar_pgvector(embedding, k=5):
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)  # Necesario para usar tipos vector en consultas
    cur = conn.cursor()

    cur.execute("""
        SELECT documento, tipo_fragmento, numero_articulo, texto
        FROM normativa
        ORDER BY embedding <-> (%s)::vector
        LIMIT %s
    """, (embedding.tolist(), k))

    resultados = cur.fetchall()
    cur.close()
    conn.close()
    return resultados

# =============================================
# 6. Generar respuesta contextual usando Ollama
# =============================================
# Esta función utiliza un modelo de lenguaje (Qwen3) para generar una respuesta.
# Usa un "prompt" que incluye el contexto legal (artículos similares encontrados)
# y la pregunta del usuario.

def responder_con_ollama(pregunta, contexto, modelo_qwen):
    prompt = f"""Eres un asistente legal experto en facturación electrónica de El Salvador.
Debes responder a la siguiente pregunta basándote en el contexto legal recuperado.
Si no hay suficiente contexto, responde de forma prudente y explícitalo.

### Contexto:
{contexto}

### Pregunta:
{pregunta}

### Respuesta:"""

    r = requests.post(OLLAMA_API, json={
        "model": modelo_qwen,
        "prompt": prompt,
        "stream": False
    })

    return r.json()["response"]

# =============================================
# 7. Pipeline principal: de la pregunta a la respuesta
# =============================================
# Este es el flujo que se ejecuta cuando el usuario hace una pregunta.
# a) Genera el embedding de la pregunta.
# b) Recupera los artículos más similares.
# c) Construye el contexto legal.
# d) Usa Qwen3 para generar la respuesta.

def consultar(pregunta):
    emb = generar_embedding(pregunta)
    fragmentos = consultar_pgvector(emb)
    contexto = "\n\n".join(f"[Art. {a or '—'}] {t}" for _, _, a, t in fragmentos)
    modelo_qwen = obtener_modelo_qwen3()
    respuesta = responder_con_ollama(pregunta, contexto, modelo_qwen)
    return respuesta.strip(), contexto

# =============================================
# 8. Interfaz Web con Gradio
# =============================================
# Gradio permite exponer funciones de Python como interfaces gráficas simples.
# En este caso, se construye una interfaz con una caja de texto para preguntas
# y dos salidas: la respuesta generada y el contexto legal utilizado.

gr.Interface(
    fn=consultar,
    inputs=gr.Textbox(label="Pregunta legal sobre facturación electrónica"),
    outputs=[
        gr.Textbox(label="Respuesta Generada"),
        gr.Textbox(label="Contexto Recuperado")
    ],
    title="Asistente Semántico Legal",
    description="Consulta normativa de El Salvador con razonamiento basado en Qwen3 y búsqueda semántica vectorial.",
).launch()
