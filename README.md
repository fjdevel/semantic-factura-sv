# Buscador Semántico de Normativa Fiscal - Facturación Electrónica en El Salvador

Implementación de un asistente legal inteligente basado en **búsqueda semántica (semantic search)** con **embeddings vectoriales** y modelos de lenguaje (LLM) para consultar la **normativa fiscal relacionada con la factura electrónica en El Salvador**.

Este proyecto combina procesamiento de lenguaje natural, PostgreSQL con pgvector, modelos de embeddings en español, e interfaces interactivas para construir una solución real con aplicaciones en entornos legales, tributarios y financieros.

> Este repositorio está diseñado para desarrolladores técnicos con interés en aplicar técnicas de inteligencia artificial (IA) al análisis de texto legal o normativo.

---

## Características principales

- 🔎 Búsqueda semántica sobre artículos legales usando embeddings
- 🧠 Generación de respuestas legales con LLM (Qwen3 vía Ollama)
- 🗃️ Base de datos PostgreSQL con índice vectorial (pgvector)
- 📄 Extracción automática de normativa desde PDFs oficiales
- 🌐 Interfaz web interactiva con Gradio
- ✅ Proyecto educativo, técnico y funcional en producción local

---

## Tecnologías utilizadas

- Python 3.10+
- PostgreSQL 15+ con extensión `pgvector`
- Modelo de embeddings en español: `jinaai/jina-embeddings-v2-base-es`
- Servidor Ollama local con modelo LLM `Qwen3`
- Framework de interfaz: Gradio
- Librerías adicionales: sentence-transformers, psycopg2, torch, pdfplumber, tqdm

---

## Casos de uso

- Automatización de consultas legales y tributarias
- Asistentes internos para cumplimiento fiscal
- Exploración semántica de documentos normativos
- Proyectos de IA aplicada a derecho, contabilidad o auditoría

---

## Requisitos previos

- Python 3.10 o superior
- PostgreSQL instalado con la extensión `pgvector`
- Ollama funcionando localmente con el modelo `qwen3` descargado
- Opcional: GPU compatible con CUDA para acelerar embeddings

---

## Instalación paso a paso

1. Clonar el repositorio:

```bash
git clone https://github.com/fjdevel/semantic-factura-sv.git
cd semantic-factura-sv
```

2. Levantar la base de datos PostgreSQL con soporte para `pgvector`:

```bash
podman-compose up -d
# o usa docker-compose si prefieres
```

3. Colocar los documentos legales (PDF) en la carpeta:

```
./data/raw/
```

4. Ejecutar el script de extracción y clasificación:

```bash
python scripts/extraccion_normativa.py
```

5. Generar embeddings e insertarlos en la base de datos:

```bash
python scripts/generar_embeddings.py
```

6. Lanzar la aplicación web:

```bash
python scripts/app_gradio.py
```

La interfaz estará disponible en:  
http://localhost:7860

---

## Ejemplo de consulta

**Pregunta:**  
¿Dónde se regula la obligación de emitir comprobantes electrónicos?

**Respuesta generada:**  
Según el artículo correspondiente, los sujetos pasivos están obligados a emitir comprobantes electrónicos conforme al sistema de factura electrónica definido por la Dirección General de Impuestos Internos...

**Fragmentos legales recuperados:**  
Art. 114, Art. 142, Guía Técnica No. 3, entre otros.

---

## Estructura del proyecto

```
semantic-factura-sv/
├── data/
│   ├── raw/                 # PDFs originales
│   └── processed/           # Fragmentos extraídos y limpios
├── db/
│   └── init.sql             # Creación de tabla, índices y extensión pgvector
├── scripts/
│   ├── extraccion_normativa.py
│   ├── generar_embeddings.py
│   └── app_gradio.py        # Interfaz web
├── docker-compose.yml
└── README.md
```

---

## ¿Qué son los embeddings y la búsqueda semántica?

Un **embedding** es un vector numérico que representa el significado de un texto. Al usar embeddings, podemos buscar textos que "significan lo mismo" aunque usen palabras diferentes.

**La búsqueda semántica** permite encontrar respuestas relevantes basadas en intención o contexto, no solo coincidencias exactas de palabras clave. En este proyecto usamos embeddings de JinaAI entrenados para español legal y técnico.

---

## Enlaces útiles

- Sitio oficial de factura electrónica: https://factura.gob.sv
- Documentación de `pgvector`: https://github.com/pgvector/pgvector
- Ollama (servidor local de LLMs): https://ollama.com/
- Modelo Qwen3: https://huggingface.co/Qwen

---

## Contribuciones

Este proyecto es de código abierto y se encuentra en fase activa de mejora. Si deseas contribuir:

1. Haz un fork
2. Crea una rama
3. Envía un Pull Request

También puedes abrir un *issue* si deseas reportar errores o sugerencias.

---

## Licencia

MIT License. Puedes usar este código libremente para fines educativos, profesionales o comerciales.

---

---

## ¿Te interesa aplicar IA práctica en desarrollo de software y sectores regulados?

Suscríbete al newsletter en LinkedIn para recibir guías, ejemplos y proyectos reales cada semana:  
[Suscribirte en LinkedIn](https://www.linkedin.com/build-relation/newsletter-follow?entityUrn=7334096493261336576)

También puedes explorar más contenido técnico y artículos en nuestro blog:  
[Visita el blog en xinmind.dev](https://www.xinmind.dev/)


