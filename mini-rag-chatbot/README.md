# Mini RAG Chatbot

A small Retrieval-Augmented Generation (RAG) chatbot built with Python, the Google GenAI SDK, Gemini embeddings, and FAISS.

## Features

- Reads a local `notes.txt` knowledge base
- Splits documents into chunks with overlap for long paragraphs
- Creates embeddings with `gemini-embedding-001`
- Stores vectors in a FAISS `IndexFlatIP` index
- Normalizes vectors so inner product acts like cosine similarity
- Retrieves the most relevant chunks for each question
- Uses `gemini-3.8-flash` to answer using retrieved context
- Prints retrieved source chunks and similarity scores
- Keeps the API key in `.env`, not in the Python source

## Requirements

- Python 3.10 or newer recommended
- A Gemini API key
- Internet access when calling the Gemini API

## Setup on Windows

Open PowerShell or the VS Code terminal in this project folder.

### 1. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure the API key

1. Get a Gemini API key from Google AI Studio: https://aistudio.google.com/apikey
2. Copy `.env.example` and name the copy `.env`.
3. Open `.env` and replace `your_gemini_api_key_here` with your real key.
4. Never share the key or commit `.env` to GitHub.

### 4. Add your knowledge

Edit `notes.txt`. Add facts or notes you want the chatbot to answer questions about. Separate paragraphs with a blank line.

### 5. Run the chatbot

```powershell
python main.py
```

Ask questions about your notes. Type `exit` or `quit` to stop.

## Example

With the sample notes, you can ask:

- What does RAG stand for?
- What is an embedding?
- Why does this project normalize vectors?
- Does RAG guarantee correct answers?

The chatbot should answer from the sample notes. For a question not covered by the notes, it should say it does not know based on the provided notes.

## How it works

1. **Load:** Read `notes.txt`.
2. **Chunk:** Split text into manageable chunks. Long paragraphs use overlapping windows.
3. **Embed:** Gemini turns each chunk into a 768-dimensional vector.
4. **Index:** FAISS stores normalized vectors.
5. **Query embedding:** The user's question is embedded with the `RETRIEVAL_QUERY` task type.
6. **Retrieve:** FAISS returns the top matching chunks.
7. **Generate:** Gemini receives the question and retrieved context and writes a grounded answer.

## Troubleshooting

- **`GEMINI_API_KEY was not found`:** Ensure the file is named `.env` (not `.env.txt`) and is beside `main.py`.
- **`ModuleNotFoundError`:** Activate the virtual environment and run `pip install -r requirements.txt`.
- **FAISS installation error on Windows:** Use a supported 64-bit Python version and upgrade pip. If installation still fails, try a fresh Python 3.11 or 3.12 virtual environment.
- **API/model/quota error:** Check that your API key is valid, the selected models are available to your account, and your API quota is not exhausted.
- **Weak answers:** Add clearer information to `notes.txt`, review the retrieved context, and adjust chunk size or `DEFAULT_TOP_K`.

## Important notes

- The first run makes embedding API calls for the knowledge base; repeated runs currently rebuild embeddings instead of caching them.
- API usage and pricing depend on the models and your account's current limits.
- This is a learning project, not a production-ready system. For larger documents, consider persistent indexes, metadata, evaluation, access control, and better document ingestion.
