# Desactivamos una opción avanzada de rendimiento que podría generar conflictos
import os
os.environ["FLASH_ATTENTION_2_ENABLED"] = "false"

# Importamos las librerías necesarias
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector 
from pathlib import Path
from tqdm import tqdm  # Para mostrar barra de progreso

# ===============================
# CONFIGURACIÓN DEL ENTORNO
# ===============================

# Nombre del modelo de embeddings preentrenado (español)
MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"

# Dimensión de los vectores generados por el modelo (por defecto es 768)
EMBEDDING_DIM = 768

# Usar GPU si está disponible, de lo contrario usar CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Datos de conexión a la base de datos PostgreSQL
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
    "dbname": "semantic_fiscal"
}

# ===============================
# CARGA DEL MODELO DE EMBEDDINGS
# ===============================

print(f"Cargando modelo {MODEL_NAME} en dispositivo: {DEVICE}")
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
    model_kwargs={"attn_implementation": "eager"}  # Para evitar errores con atención acelerada
)

# ===============================
# CARGA DEL DATASET DE NORMATIVA
# ===============================

# Cargamos el archivo CSV que contiene los fragmentos legales ya extraídos
df = pd.read_csv("./data/processed/fragmentos_explorados.csv")

# ===============================
# FUNCIÓN PARA GENERAR EMBEDDINGS
# ===============================

def generar_embedding(texto):
    """
    Recibe un fragmento de texto y devuelve su representación como vector semántico (embedding).
    Normaliza el vector para que tenga magnitud 1 (recomendado para búsquedas con distancia angular).
    """
    emb = model.encode(texto, normalize_embeddings=True)
    return emb[:EMBEDDING_DIM].tolist()  # Asegurarse de que la dimensión sea la esperada

# ===============================
# APLICAR EMBEDDINGS A CADA FRAGMENTO
# ===============================

print("Generando embeddings para cada fragmento legal...")
df["embedding"] = df["texto"].apply(generar_embedding)

# ===============================
# CONEXIÓN A LA BASE DE DATOS
# ===============================

print("🔌 Conectando a PostgreSQL...")
conn = psycopg2.connect(**DB_CONFIG)

# Registramos el tipo especial "vector" en psycopg2 para poder insertar embeddings
register_vector(conn)

cur = conn.cursor()

# ===============================
# INSERCIÓN DE DATOS EN LA TABLA
# ===============================

print("Insertando registros en la tabla 'normativa'...")

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

# Guardamos los cambios
conn.commit()

# Cerramos la conexión
cur.close()
conn.close()

print("Proceso completado con éxito.")
