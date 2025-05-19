# 🔍 AI-Powered Financial Q&A Bot with RAG and Dual Retrieval

This project is an **AI-powered Q&A chatbot** that answers questions based on 10 quarterly financial PDFs. It uses a combination of **structured** (SQLite) and **unstructured** (vector DB) data to provide accurate, flexible responses. It also supports **trend plots** and **natural language querying** using OpenAI’s GPT-3.5 Turbo.

---

## 📌 Objective

To develop a chatbot capable of:
- Extracting financial metrics into a SQL table (`metric_table`)
- Extracting insights and image captions into a vector database for retrieval
- Supporting both **unstructured** and **structured** data queries
- Providing **trend line plots** for metrics over quarters
- Allowing users to **interact through a Gradio interface**

---

## 🚀 Key Features

- ✅ Dual-retrieval pipeline (RAG + SQLite)
- ✅ GPT-3.5-based natural language response
- ✅ Supports trend chart generation (matplotlib)
- ✅ Real-time interaction via Gradio UI
- ✅ Seamless Hugging Face deployment
- ✅ Metric table editor for add/edit/delete metrics

---

## ⚙️ Technologies Used

- `Python`
- `Gradio`
- `OpenAI GPT-3.5 Turbo`
- `FAISS` for vector similarity search
- `SentenceTransformers` for embeddings
- `SQLite` for structured metric storage
- `Pandas`, `Matplotlib`, `Regex` for data processing

---

## 🧠 Project Architecture

```
PDFs (10) ──► Extract Text & Images ──► Chunked + Captioned Text
                                │
                                ├──► Vector Embedding (FAISS)
                                └──► Financial Metric Extraction (SQLite)

User Query ──► Semantic Match (FAISS) + SQL Match (if metric) ──► GPT-3.5 Prompt ──► Answer + Optional Plot
```

---

## 📂 Folder Structure

```
├── app.py                      # Main Gradio app for deployment
├── faiss_index.index           # FAISS index of vector chunks
├── faiss_metadata.pkl          # Metadata for each FAISS chunk
├── financial_metrics.db        # SQLite DB with revenue, EBITDA, PAT
├── sample_Q1FY25.pdf           # Example financial PDF
├── requirements.txt            # All required Python packages
└── README.md                   # This file
```

---

## 📥 How to Run Locally

1. Clone this repo:
```bash
git clone https://github.com/tgchandu25/RAG-LLM-Financial-QA-Bot.git
cd RAG-LLM-Financial-QA-Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=your-api-key-here
```

4. Run the app:
```bash
python app.py
```

---

## 🌐 Live Hugging Face Deployment

Visit the deployed chatbot here:  
➡️ https://huggingface.co/spaces/TGChandu/RAG_LLM_Financial_QA_Chatbot

---

## 🧪 Example Questions to Try

- “What is the trend in net profit over the last 8 quarters?”
- “How has the EBITDA margin changed since Q2FY23?”
- “What are the growth projections in Q1FY25?”
- “What did the CEO mention about competitive threats?”

---

## 🛠️ Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| Extracting info from images | Used PaddleOCR for visual text extraction |
| Answering both metric and paragraph queries | Integrated dual-retrieval pipeline (RAG + SQL) |
| Ensuring Hugging Face compatibility | Avoided file path issues, used relative paths |

---

## 📌 Conclusion

This project showcases a complete, production-grade chatbot capable of handling nuanced financial queries using both structured and unstructured data. Its hybrid architecture ensures versatility in real-world reporting or investor Q&A scenarios. The deployment on Hugging Face enables easy accessibility for demo and testing.

---

## 👨‍💻 Author

Developed and submitted as part of an internship assessment.  
Project by **TG Chandu**.
