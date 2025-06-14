# 🧠 Neurona

Neurona is an AI-powered semantic memory engine designed to extract, embed, and search insights from PDF documents. It offers a seamless interface to upload documents, perform advanced text embedding using Sentence Transformers, and retrieve the most relevant semantic results in real time.

> Built with cutting-edge NLP tools, modern full-stack tech, and a polished UI — **Neurona is market-ready and showcase-worthy.**

---

## 🚀 Features

- 📄 **PDF Upload** with automatic parsing and chunking
- 🧠 **AI Embedding** using `all-mpnet-base-v2` via SentenceTransformers
- 🔍 **Semantic Search** powered by FAISS for similarity-based ranking
- ⚡️ **FastAPI backend** for embedding and querying
- 💻 **Next.js + Tailwind + ShadCN UI** frontend with beautiful animations
- 🌐 **Cross-origin support** (CORS enabled)
- 🎯 Designed for real-world deployment & scale

---

## 🧰 Tech Stack

| Layer         | Tools Used                                                                 |
|---------------|----------------------------------------------------------------------------|
| **Frontend**  | Next.js, TypeScript, Tailwind CSS, ShadCN UI, Framer Motion                |
| **Backend**   | FastAPI, Python 3.10+, FAISS, PyMuPDF, NLTK                                |
| **Embeddings**| SentenceTransformers (`all-mpnet-base-v2`, `MiniLM`, `DistilUSE`)         |
| **Storage**   | JSON-based vector persistence for simplicity                               |
| **Infra**     | Local development using `uvicorn`, CORS for API communication              |

---

## 🧪 Local Setup

### 🔧 Backend (FastAPI)

```bash
# 1. Clone and go into the repo
git clone https://github.com/prafullpandeyy/neurona.git
cd neurona

# 2. Setup virtual environment
python -m venv venv
.\venv\Scripts\activate  # (Windows) or source venv/bin/activate (Mac/Linux)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run backend server
uvicorn main:app --reload --port 8000

💻 Frontend (Next.js + Tailwind)
bash
Copy code
# In another terminal
cd neurona-ui

# Install packages
npm install

# Start the dev server
npm run dev
Visit: http://localhost:3000

📁 Folder Structure
bash
Copy code
neurona/
│
├── app/                     # Core backend modules
│   ├── parser.py            # PDF parsing logic (via PyMuPDF)
│   ├── chunker.py           # Chunking logic for documents
│   ├── embedder.py          # Model loading, embedding, query
│   ├── vector_store.py      # FAISS index creation/search
│   ├── search.py            # NeuronaSearchEngine abstraction
│
├── data/
│   ├── uploads/             # Uploaded PDF files
│   └── embeddings/          # JSON saved vector data
│
├── neurona-ui/              # Frontend app
│   ├── app/page.tsx         # UI with upload & semantic search
│   └── components/          # ShadCN UI components
│
├── main.py                  # FastAPI routes (/embed, /search)
├── requirements.txt
└── README.md
📷 UI Preview
Upload PDF	Semantic Search

(Replace with your own screenshots once deployed)

📌 TODO & Future Enhancements
 PDF summarization using LLM (e.g., GPT-4 Turbo)

 User authentication & document history

 Multi-PDF memory store (per user or project)

 Deployment on Vercel (frontend) + Render (backend)

 Add embedding drift detection via cosine distance

🤝 Contact / Contributions
Created by Prafull Pandey
For collaboration, ideas, or mentorship: prafullp41@gmail.com

⚡ Neurona bridges the gap between raw data and real understanding — fast, accurate, and beautifully designed.
