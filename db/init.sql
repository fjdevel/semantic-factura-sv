CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS normativa (
  id SERIAL PRIMARY KEY,
  documento TEXT NOT NULL,
  tipo_fragmento TEXT,
  numero_articulo INTEGER,
  texto TEXT NOT NULL,
  embedding vector(768),
  fecha_ingreso TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índice para búsquedas semánticas
CREATE INDEX IF NOT EXISTS idx_norm_act_embedding
  ON normativa
  USING ivfflat (embedding vector_l2_ops)
  WITH (lists = 100);

-- Índices clásicos
CREATE INDEX IF NOT EXISTS idx_normativa_documento ON normativa (documento);
CREATE INDEX IF NOT EXISTS idx_normativa_tipo ON normativa (tipo_fragmento);
CREATE INDEX IF NOT EXISTS idx_normativa_articulo ON normativa (numero_articulo);