import os
os.environ["FLASH_ATTENTION_2_ENABLED"] = "false"

from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector 
from pathlib import Path
from tqdm import tqdm

# Configuración
MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"
EMBEDDING_DIM = 768

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
    "dbname": "semantic_fiscal"
}

# Cargar modelo
print(f"Cargando modelo {MODEL_NAME} en {DEVICE}")
model = SentenceTransformer(MODEL_NAME, device=DEVICE, model_kwargs={"attn_implementation": "eager"})


# Leer dataset
df = pd.read_csv("./data/processed/fragmentos_explorados.csv")

# Función de generación
def generar_embedding(texto):
    emb = model.encode(texto, normalize_embeddings=True)
    return emb[:EMBEDDING_DIM].tolist()

# Aplicar embeddings
print("Generando embeddings...")
df["embedding"] = df["texto"].apply(generar_embedding)

# Conectar a PostgreSQL
print("Conectando a PostgreSQL...")
conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)
cur = conn.cursor()

# Insertar en tabla
print("Insertando en base de datos...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    cur.execute("""
        INSERT INTO normativa (documento, tipo_fragmento, numero_articulo, texto, embedding)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        row["documento"],
        row["tipo_fragmento"],
        int(row["numero_articulo"]) if not pd.isna(row["numero_articulo"]) else None,
        row["texto"],
        row["embedding"]
    ))

conn.commit()
cur.close()
conn.close()
print("✅ Proceso completado con éxito.")
