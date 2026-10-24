# Armin — Your Organizational SOP Assistant

Armin is a local AI-powered chatbot designed to help users find answers in organizational Standard Operating Procedures (SOPs).

## Features

- Ask questions about SOP documents using a chat interface.
- Retrieve relevant SOP content using semantic search.
- Generate answers using a locally hosted language model.
- Use a React frontend and FastAPI backend.
- Refresh the semantic index when documents change.

## Technology Stack

- **Frontend:** React, Vite
- **Backend:** Python, FastAPI
- **Language model:** Gemma 3:4B through Ollama
- **Embeddings:** `nomic-embed-text`
- **Semantic search:** ChromaDB
- **PDF processing:** PyMuPDF

## Prerequisites

Install the following before running Armin:

- Python
- Node.js and npm
- Ollama

Pull the required Ollama models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

Model availability and naming may vary by Ollama version.

## Project Structure

```text
sop-chatbot/
├── backend/
│   ├── main.py
│   ├── test_sop_index.py
│   └── test_sop_search.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/LanceLeot/armin-sop-assistant.git
cd armin-sop-assistant
```

### 2. Set up the backend

From the project root, open a terminal and run:

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies required by the project. Make sure the necessary Python packages are installed before starting the server.

### 3. Add sample SOP documents

Create a `backend/documents` folder and place your authorized sample PDF documents inside it.

The documents and local semantic database are excluded from Git, so they must be supplied locally.

### 4. Set up the frontend

Open a second terminal from the project root:

```bash
cd frontend
npm install
```

### 5. Run Armin

Start Ollama and ensure the required models are available. Activate the backend virtual environment, then start the API:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload
```

Follow the frontend development instructions in `frontend/README.md` if you want to run the Vite development server separately.

## Privacy and Data

Armin is designed to use a local language model. Actual privacy depends on the model configuration, application code, and any external services used.

Do not add credentials, confidential organizational documents, or private data to this public repository.

## Project Status

This project is under development. Setup details and features may change as development continues.
