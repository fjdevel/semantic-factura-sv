import os
import gradio as gr
import psycopg2
import numpy as np
import torch
import requests
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# Configuraciones
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
    "dbname": "semantic_fiscal"
}
EMBEDDING_DIM = 768
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-es"
OLLAMA_API = "http://localhost:11434/api/generate"

# Cargar modelo de embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(EMBEDDING_MODEL, device=device)

# Detectar modelo Qwen3 disponible en Ollama
def obtener_modelo_qwen3():
    modelos = requests.get("http://localhost:11434/api/tags").json().get("models", [])
    for m in modelos:
        if "qwen3" in m["name"].lower():
            return m["name"]
    raise RuntimeError("Qwen3 no encontrado en Ollama")

# Función para generar embeddings
def generar_embedding(texto):
    emb = model.encode(texto, normalize_embeddings=True)
    return emb[:EMBEDDING_DIM]

# Consultar fragmentos similares en PostgreSQL
def consultar_pgvector(embedding, k=5):
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
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

# Generar respuesta con Ollama
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

# Pipeline
def consultar(pregunta):
    emb = generar_embedding(pregunta)
    fragmentos = consultar_pgvector(emb)
    contexto = "\n\n".join(f"[Art. {a or '—'}] {t}" for _, _, a, t in fragmentos)
    modelo_qwen = obtener_modelo_qwen3()
    respuesta = responder_con_ollama(pregunta, contexto, modelo_qwen)
    return respuesta.strip(), contexto

# Interfaz Gradio
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
