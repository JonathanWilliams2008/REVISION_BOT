# 📘 A-Level Economics AI Chatbot (Edexcel Spec)

This is an AI-powered study chatbot built to help with **Edexcel A-Level Economics A (9EC0)**. It answers questions, explains key concepts, generates exam-style questions, and gives model answers — all based on official spec documents, formula sheets, past papers, and more.

---

## ⚙️ Features

- Answers questions using real Edexcel materials
- Understands exam techniques (definitions, model answers, analysis, etc.)
- Gives short, exam-style responses (or full answers if asked)
- Uses AI to read your PDFs and OCR images
- Cites sources from your uploaded files
- Simple chat interface built with Gradio

---

## 🛠️ How to Run It

### 1. Clone the project & install requirements

``
git clone https://github.com/your-username/econ-a-level-chatbot.git
cd econ-a-level-chatbot
pip install -r requirements.txt
2. Add your API key
Create a .env file in the root folder and add:


Copy
Edit
API_KEY=your_together_ai_key_here
Get a free key from https://together.ai

3. Add your documents
Put PDFs, notes, and images into the data/ folder. The chatbot uses them to generate accurate answers.

4. Run the chatbot

Copy
Edit
python app.py
Then open http://localhost:7860 in your browser.

💡 Tech Stuff
Uses FAISS for vector search

Mistral 7B model via Together AI

Gradio for the frontend

OCR support for scanned PDFs and images

Built with LangChain + HuggingFace embeddings

📚 Example Use Cases
"What is price elasticity of demand?"

"Give me a 25-mark question on inflation"

"Explain a negative output gap"

"Give a model answer for this question..."

🧠 Made By
Jonathan Williams – A-Level student (2025)
Built as a revision tool to make studying smarter, not harder.
