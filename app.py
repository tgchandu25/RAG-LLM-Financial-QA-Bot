import gradio as gr
import openai
import faiss
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer

# Paths (local for Hugging Face deployment) 
FAISS_INDEX_PATH = "faiss_index.index"
FAISS_METADATA_PATH = "faiss_metadata.pkl"
SQLITE_DB_PATH = "financial_metrics.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#  Initialize OpenAI client 
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

#  Load FAISS index 
index = faiss.read_index(FAISS_INDEX_PATH)

#  Load metadata 
with open(FAISS_METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

#  SentenceTransformer Model 
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#  Helper: Load DB fresh per request 
def get_metric_df():
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    df = pd.read_sql("SELECT * FROM metric_table", conn)
    conn.close()
    return df

#  Trend Detection 
def extract_metric_from_question(q):
    keywords = {
        "net profit": "pat",
        "pat": "pat",
        "revenue": "revenue",
        "ebitda": "ebitda",
        "operating margin": "ebitda_margin",
        "pat margin": "pat_margin"
    }
    q = q.lower()
    for key in keywords:
        if key in q and "trend" in q:
            return keywords[key]
    return None

def plot_metric_trend(metric):
    df = get_metric_df()
    df = df.dropna(subset=[metric])
    df = df.sort_values("quarter")
    plt.figure(figsize=(8, 4))
    plt.plot(df["quarter"], df[metric], marker="o")
    plt.title(f"{metric.replace('_', ' ').title()} Trend")
    plt.xlabel("Quarter")
    plt.ylabel(metric.replace("_", " ").title() + " (₹ Cr)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = "trend_chart.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

#  Answer Generator 
def answer_question(question):
    df = get_metric_df()
    metric = extract_metric_from_question(question)
    chart = None

    # Vector DB retrieval
    query_vector = embedder.encode([question])
    D, I = index.search(np.array(query_vector), k=5)
    top_chunks = [metadata[i] for i in I[0] if i < len(metadata)]
    context_text = "\n".join(top_chunks)

    # Structured metrics as inline context
    metric_summary = df.to_markdown(index=False)

    # GPT Prompt
    prompt = f"""
You are a financial assistant bot. Use the provided context and metric table to answer the question.

Context:
{context_text}

Metric Table:
{metric_summary}

Question: {question}
Answer:
"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()

    # Add chart if trend-related
    if metric:
        chart = plot_metric_trend(metric)

    return answer, chart

#  Gradio UI 
def handle_submit(q):
    try:
        answer, img_path = answer_question(q)
        img = img_path if img_path else None
        return answer, img
    except Exception as e:
        return f"Error: {str(e)}", None

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("📊 **Financial Chatbot**")
    gr.Markdown("Ask questions about metrics, quarters, trends or CEO commentary.")

    with gr.Row():
        question_input = gr.Textbox(label="Your Question", placeholder="e.g. What is the trend in EBITDA?", lines=2)
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer", lines=10)
    with gr.Row():
        image_output = gr.Image(label="📈 Trend Chart", type="filepath")

    submit_btn.click(fn=handle_submit, inputs=question_input, outputs=[answer_output, image_output])
    clear_btn.click(lambda: ("", None), outputs=[answer_output, image_output])

demo.launch()