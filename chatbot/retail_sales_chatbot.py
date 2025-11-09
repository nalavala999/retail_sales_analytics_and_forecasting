import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import requests
from databricks import sql

# ------------------------------------------------------------
# 🧠 1. App Title
# ------------------------------------------------------------
st.title("🛍️ Retail Sales Analytics Chatbot (Powered by Grok-3)")

# ------------------------------------------------------------
# 🔐 2. Load credentials from .env
# ------------------------------------------------------------
load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_MODEL = "grok-3"

DATABRICKS_SERVER = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_ACCESS_TOKEN")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 1000

# ------------------------------------------------------------
# 🧩 3. Connect to Databricks
# ------------------------------------------------------------
def get_conn():
    return sql.connect(
        server_hostname=DATABRICKS_SERVER,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )

# ------------------------------------------------------------
# 📦 4. Fetch data from your Databricks retail table
# ------------------------------------------------------------
def fetch_sales_data(limit=5000):
    query = f"""
    SELECT 
      f.order_id,
      o.order_date,
      r.region,
      p.category,
      p.sub_category,
      f.sales,
      f.profit,
      f.quantity,
      f.discount,
      o.ship_mode,
      c.segment
    FROM retail_sales_analytics_and_forecasting.gold.fact_sales f
    JOIN retail_sales_analytics_and_forecasting.gold.dim_region r 
      ON f.region_sk = r.region_sk
    JOIN retail_sales_analytics_and_forecasting.gold.dim_product p 
      ON f.product_sk = p.product_sk
    JOIN retail_sales_analytics_and_forecasting.gold.dim_customer c 
      ON f.customer_sk = c.customer_sk
    JOIN retail_sales_analytics_and_forecasting.gold.dim_order o 
      ON f.order_sk = o.order_sk
    LIMIT {limit};
    """
    with get_conn() as conn:
        df = pd.read_sql(query, conn)
    return df


# ------------------------------------------------------------
# 🧾 5. Convert data to text documents
# ------------------------------------------------------------
def df_to_documents(df: pd.DataFrame):
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Order {row['order_id']} placed on {row['order_date']} "
            f"in region {row['region']} under category {row['category']} ({row['sub_category']}). "
            f"Sales amount: {row['sales']}, Profit: {row['profit']}, Quantity: {row['quantity']}, Discount: {row['discount']}. "
            f"Shipped via {row['ship_mode']} to a {row['segment']} customer."
        )
        docs.append(text)
    return docs

# ------------------------------------------------------------
# 🧩 6. Chunking
# ------------------------------------------------------------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for d in docs:
        chunks.extend(splitter.split_text(d))
    return chunks

# ------------------------------------------------------------
# 🔍 7. Embedding & FAISS index
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index_batched(chunks, batch_size=EMBED_BATCH_SIZE):
    model = load_embed_model()
    index = None
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch = chunks[start:end]
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
        index.add(embs)
    return index, model

def search(index, model, chunks, query, k=30):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(np.array(q), k)
    return "\n".join([chunks[i] for i in I[0] if 0 <= i < len(chunks)])

# ------------------------------------------------------------
# 🤖 8. Grok API call
# ------------------------------------------------------------
def ask_xai(context, question):
    if not XAI_API_KEY:
        st.error("XAI_API_KEY missing")
        return None

    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        "You are a retail sales analytics assistant. "
        "Answer strictly using the context below. If information is missing, say 'I don’t know.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    data = {
        "model": XAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a data analyst specialized in retail sales and forecasting."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    if not resp.ok:
        st.error(f"x.ai API error {resp.status_code}: {resp.text}")
        return None
    return resp.json()["choices"][0]["message"]["content"]

# ------------------------------------------------------------
# 🖥️ 9. Streamlit UI
# ------------------------------------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.model = None

if st.button("📊 Build Retail Index from Databricks"):
    with st.spinner("Loading data from Databricks and building index..."):
        df = fetch_sales_data(limit=5000)
        docs = df_to_documents(df)
        chunks = chunk_documents(docs)
        index, model = build_index_batched(chunks)
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.model = model
        st.success(f"✅ Loaded {len(docs)} orders → {len(chunks)} chunks → FAISS index ready.")

question = st.text_input("💬 Ask about sales, profit, customers, or regions:")

if question and st.session_state.index is not None:
    with st.spinner("Thinking..."):
        ctx = search(st.session_state.index, st.session_state.model, st.session_state.chunks, question)
        ans = ask_xai(ctx, question)
        st.markdown("### 🤖 Answer")
        st.write(ans)
        with st.expander("📄 Context Used"):
            st.write(ctx)
