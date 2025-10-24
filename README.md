<h1 align="center">✨ AI Engineer Portfolio — Sreeja Akuthota ✨</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=900&center=true&vCenter=true&width=850&lines=AI+Resume+%26+Cover+Letter+Generator+%F0%9F%93%9D%E2%9A%A1;Gen+AI+PDF+Q%26A+Chatbot+%F0%9F%93%96%E2%80%8D%F0%9F%92%BB;AI+Review+Sentiment+Dashboard+%F0%9F%8E%89%2F%F0%9F%93%8A;NLP+%7C+RAG+%7C+LLM+Apps+in+Production" />
</p>

<p align="center">
  <a href="mailto:sreejaakuthota07@gmail.com"><img src="https://img.shields.io/badge/Email-Contact-informational?logo=gmail"></a>
  <a href="https://www.linkedin.com/in/sreeja-akuthota/"><img src="https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin"></a>
  <img src="https://img.shields.io/badge/Role-AI%20Engineer-purple">
  <img src="https://img.shields.io/badge/Location-TX%2C%20USA-lightgrey">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/HuggingFace-ffd21e?logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/LangChain-0891b2?logo=chainlink&logoColor=white">
  <img src="https://img.shields.io/badge/Vector%20DB-FAISS%2FChroma-success">
  <img src="https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Spaces-Hugging%20Face-yellow?logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/AWS%20SageMaker-orange?logo=amazonaws&logoColor=white">
  <img src="https://img.shields.io/badge/Azure%20ML-0078D4?logo=microsoftazure&logoColor=white">
  <img src="https://img.shields.io/badge/GCP%20Vertex%20AI-4285F4?logo=googlecloud&logoColor=white">
  <img src="https://img.shields.io/badge/SQL-Postgres%20%7C%20MySQL-blue?logo=postgresql&logoColor=white">
  <img src="https://img.shields.io/badge/BI-Tableau%20%7C%20Power%20BI-yellow?logo=tableau&logoColor=white">
</p>

---

## 🚀 Featured Projects

<details open>
<summary><strong>Project 1 — AI Resume & Cover Letter Generator</strong> · <em>NLP</em></summary>

**Live:** <a href="https://ai-resume-coverletter-generator.streamlit.app/">Streamlit Cloud</a>  
**Key Skills:** Prompt engineering, LLM text generation  
**Stack:** Python · Streamlit · FastAPI (optional API) · OpenAI/HF Inference API · Prompt Templates

<p align="center">
  <img src="dashboard.png" alt="AI resume generator" width="72%" style="border-radius:12px"/>
  <br/><img src="pulse.gif" width="180" alt="pulse"/>
</p>

**Highlights**
- One-click generation of tailored resume bullets and cover letters from a job description.
- Guardrails & prompt-chains for tone, length, and ATS keyword coverage.
- Export to PDF/Markdown; reusable prompt presets per role.


<details open>
<summary><strong>Project 2 — Gen AI PDF Q&A Chatbot</strong> · <em>RAG / LangChain</em></summary>

**Live:** <a href="https://huggingface.co/spaces/Sreeja007/genai-pdf-qa-chatbot-groq">Hugging Face Spaces</a>  
**Key Skills:** Embeddings, vector database, retrieval  
**Stack:** Python · LangChain · FAISS/Chroma · Hugging Face · Streamlit/Gradio

<p align="center">
  <img src="PDFChatbot.png" alt="PDF Q&A" width="72%" style="border-radius:12px"/>
  <br/><img src="pulse.gif" width="180" alt="pulse"/>
</p>

**Highlights**
- Drop PDFs and chat with them using contextual retrieval.
- Persistent vector store; sources with confidence scores.
- Chunking + re-ranking; streaming responses w/ citations.

```mermaid
sequenceDiagram
  participant U as User
  participant UI as Web UI
  participant VS as Vector DB
  participant LLM as LLM
  U->>UI: Ask question about PDF(s)
  UI->>VS: Retrieve top-k chunks (similarity)
  VS-->>UI: Relevant passages + scores
  UI->>LLM: Compose w/ retrieved context
  LLM-->>UI: Answer + inline citations
```

**Run locally**
```bash
pip install -r requirements.txt
python -m scripts/ingest --path data/pdfs
streamlit run apps/pdf_chat.py
```
</details>

---

<details open>
<summary><strong>Project 3 — AI Review Sentiment Dashboard</strong> · <em>Data + AI</em></summary>

**Live:** <a href="https://ai-review-sentiment-dashboard.streamlit.app/">Streamlit Cloud (replace link)</a>  
**Key Skills:** NLP analysis, data visualization  
**Stack:** Python · Pandas · scikit-learn/HF · Streamlit · Plotly · Power BI/Tableau (optional)

<p align="center">
  <img src="GenAI_pdf_reader.png" alt="Sentiment dashboard" width="72%" style="border-radius:12px"/>
  <br/><img src="pulse.gif" width="180" alt="pulse"/>
</p>

**Highlights**
- Ingest CSV/JSON reviews; classify sentiment and key themes.
- Trend lines, word clouds, and cohort breakdowns.
- Export annotated dataset and executive-ready charts.

```mermaid
flowchart TD
  S[Review Data] --> C[Clean + Tokenize]
  C --> M[Sentiment + Topic Models]
  M --> V[Streamlit Viz + KPIs]
  V --> X[CSV/PNG/PDF Exports]
```

**Run locally**
```bash
pip install -r requirements.txt
streamlit run apps/sentiment_dashboard.py
```
</details>

---

## ✨ Demos (Add your GIFs)
Put short demos in `assets/` and reference them here.
<p align="center">
  <img src="assets/demo_resume.gif" alt="Resume Generator Demo" width="85%"><br/>
  <img src="assets/demo_pdfqa.gif" alt="PDF Q&A Demo" width="85%"><br/>
  <img src="assets/demo_sentiment.gif" alt="Sentiment Dashboard Demo" width="85%">
</p>

---

## 👩🏽‍💻 About Me
AI Engineer with 5+ years building ML/NLP products end‑to‑end—data → models → cloud deployment → dashboards. Experience spans healthcare and insurance, productionizing models on AWS SageMaker, Azure ML, and Vertex AI; building RAG workflows; and shipping BI dashboards that drive decisions. Highlights:
- +22% NLP accuracy with HF Transformers on SageMaker.
- −20% inference latency via Dockerized PyTorch + FastAPI.
- −35% manual reporting with automated BI + anomaly detection.
- +40% dataset accuracy through ML‑validated SQL pipelines.

> Email: <a href="mailto:sreejaakuthota07@gmail.com">sreejaakuthota07@gmail.com</a> · 
> LinkedIn: <a href="https://www.linkedin.com/in/sreeja-akuthota/">sreeja-akuthota</a>

---

## 🧱 Repo Layout (suggested)
```
.
├── apps/
│   ├── resume_app.py
│   ├── pdf_chat.py
│   └── sentiment_dashboard.py
├── scripts/
│   └── ingest.py
├── requirements.txt
├── assets/
│   ├── demo_resume.gif
│   ├── demo_pdfqa.gif
│   └── demo_sentiment.gif
└── README.md
```

---

## ⚙️ Quick Setup
```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
# 2) Install deps
pip install -r requirements.txt
# 3) (Optional) set API keys
export OPENAI_API_KEY=...         # or HUGGINGFACEHUB_API_TOKEN=...
export LANGCHAIN_TRACING_V2=true  # if you want observability
```

**requirements.txt (suggested)**
```
streamlit
fastapi
uvicorn
langchain
faiss-cpu
chromadb
pydantic
python-dotenv
pandas
numpy
scikit-learn
matplotlib
plotly
transformers
torch
sentence-transformers
tabulate
```

---

## 🧪 Test & Quality
- Unit tests for chunking, retrieval, and prompt templates.
- Smoke tests for APIs (`uvicorn apps:app --reload`).
- Lint/format: `ruff`, `black`, `mypy` (optional).

---

## ☁️ Deploy
- **Streamlit Cloud**: Push repo → New app → Select `apps/resume_app.py` or `apps/sentiment_dashboard.py`.
- **Hugging Face Spaces**: Space → Gradio/Streamlit → set `requirements.txt` + `HF_TOKEN`.
- **Containers**: `docker build -t ai-portfolio . && docker run -p 8501:8501 ai-portfolio`.

---


## 🏆 Activity & Stats
<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=SreejaAkuthota&show_icons=true&theme=radical&hide_border=true" height="160" />
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=SreejaAkuthota&layout=compact&theme=radical&hide_border=true" height="160" />
</p>
<p align="center">
  <img src="https://github-readme-streak-stats-eight.vercel.app?user=SreejaAkuthota&theme=radical&hide_border=true" />
</p>

<p align="center">
  <img src="https://github-profile-trophy.vercel.app/?username=SreejaAkuthota&theme=dracula&no-frame=true&margin-w=12&row=1" />
</p>

