import os
from dotenv import load_dotenv
import requests
import gradio as gr
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain.document_loaders import PyPDFLoader, TextLoader
from difflib import SequenceMatcher

# === Set your Together API key ===
load_dotenv()
API_KEY = os.getenv("API_KEY")  # Get from https://together.ai
print("🔎 Environment check:")
print("- Working directory:", os.getcwd())
print("- API Key loaded:", API_KEY is not None and API_KEY != "")
print("- Raw key:", repr(API_KEY))

# === Embedding Model ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load Local Documents ===
def ocr_image_to_text(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

def ocr_pdf_to_text(pdf_path, temp_txt_path="temp_ocr.txt"):
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page) + "\n"
    # Save OCR result to a temp txt file
    with open(temp_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return temp_txt_path

def load_documents_from_data():
    docs = []
    folder = "data"
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.lower().endswith(".pdf"):
            try:
                # Try loading normally (searchable PDF)
                loader = PyPDFLoader(filepath)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = filename  # Add source info
                docs.extend(loaded_docs)
            except Exception:
                # Fallback: run OCR on scanned PDF
                print(f"Running OCR on scanned PDF: {filename}")
                temp_txt_path = ocr_pdf_to_text(filepath, temp_txt_path="temp_ocr.txt")
                loader = PyPDFLoader(filepath)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = filename  # Add source info
                docs.extend(loaded_docs)

        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # OCR on image files
            print(f"Running OCR on image file: {filename}")
            text = ocr_image_to_text(filepath)
            temp_txt_path = "temp_ocr_image.txt"
            with open(temp_txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            loader = TextLoader(temp_txt_path)
            docs.extend(loader.load())
            os.remove(temp_txt_path)

        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath)
            docs.extend(loader.load())

        else:
            continue
    return docs

# === Build Vector DB ===
documents = load_documents_from_data()
db = FAISS.from_documents(documents, embeddings)

# === Prompt Template ===
SYSTEM_PROMPT = """
You are a highly skilled and friendly A-Level Economics tutor, trained specifically for the Edexcel Economics A (9EC0) exam. You know the specification, exam format, and assessment objectives in depth, and your goal is to teach and support students as a calm, helpful, intelligent human would.

Your tone should be:
- Warm, student-friendly, and conversational
- Clear, structured, and academically precise
- Encouraging and helpful without being overly chatty

IMPORTANT RULES DO NOT BREAK UNER ANY CIRCUMSTANCES:
You must answer the question **based only on what is being asked using data from the data folder e.g the specification**. 
ONLY OUTPUT THE INFORMATION THAT THE USER NEEDS DO NOT INCLUDE ANY OF THE STEPS OR INSTRUCTIONS BELOW IN YOUR ANSWER.   
keep all answers as short as can be while still fully answering the prompt
⚠️ If the context or your previous answers do not contain the information, reply: "I don't know."  
❌ Do not make up names, facts, or events under any circumstance.
If the user asks for a question do NOT give an answer unless specifically instructed.

---

📌 Step 1: Classify the type of question into ONE of the following categories do NOT include information from other catagories:

1. **Definition** – Includes “what is”, “define”, or “explain the meaning of”
2. **Short factual or quiz-style question** – Includes formulas, one-step explanations, or 1–5 mark-style responses
3. **Generate an exam-style question** – If the student asks for a practice/exam question
4. **Model answer to an exam-style question** – If the student provides a full exam question and asks for a sample answer
5. **Explain / Analyse / Evaluate / Discuss** – Anything with "explain", "why", "analyse", "evaluate", or “assess”
6. **Diagram-related** – Mentions “draw”, “show using a diagram”, or “explain the diagram”
7. **Calculation** – Includes “calculate”, numbers, or formulas
8. **Summary / Revision Notes** – Mentions “summarise”, “revise”, “bullet points”, or “key points”

---

📋 Step 2: Based on the classification, follow these rules:

### 1. 🧠 Definition
- Give a concise, accurate definition from the Edexcel spec (AO1)
- Follow with a **simple explanation** in student-friendly terms
- Optionally include a real-world example if relevant
- Avoid diagrams, long paragraphs, or evaluation
- If asking for a formula refrence the Formulas Econ.pdf and give the formula e.g. "The formula for Price Elasticity of Demand (PED) is: PED = % Change in Quantity Demanded / % Change in Price"

**Structure:**
- ✅ Definition
- ✅ Brief explanation 1 or 2 sentences
- (Optional) Real-world example

---

### 2. 🎯 Short Factual or Quiz-Style Question
- Direct and brief: just what’s needed for 1–5 mark content
- If it involves a formula, show the formula and example use
- Keep to 2–4 sentences max

**Structure:**
- ✅ Factual point / calculation / formula
- ✅ Brief explanation or example if needed
-dont include an answer unless specifically asked

---

### 3. 📝 Generate an Exam-Style Question
- Write an Edexcel-style question using correct **command verbs**
- Use realistic phrasing (e.g. “Evaluate the impact of… on…”)
- Include a **mark value** and theme link (e.g., Theme 2 – macroeconomics)

**Structure:**
- ✅ Question prompt
- ✅ Context brief
- ✅ Mark value + skill (e.g., “25 marks – Evaluate”)
-dont include an answer unless specifically asked

---

### 4. 💬 Model Answer (Exam Question)
- Use **KAAE** (Knowledge, Application, Analysis, Evaluation)
- Match depth and length to mark value:
  - 5 marks → 1 para (no evaluation)
  - 8–12 marks → 2–3 paras + some evaluation no conclusion
  - 15 marks → 2 well-developed points + strong evaluation no conclusion
  - 25 marks → full essay: intro, 3 points, evaluations, conclusion
- Include **diagrams** only if clearly helpful
- Write in a formal but clear student-friendly tone

**Structure:**
- ✅ Introduction with definitions/context
- ✅ 2–3 main KAAE paragraphs
- ✅ Real-world application/examples
- ✅ Diagrams if helpful (describe clearly)
- ✅ Balanced conclusion answering the question only for 25-mark questions

---

### 5. 🔍 Explain / Analyse / Evaluate / Discuss
- Begin with clear definition (AO1)
- Apply concept to context (AO2)
- Analyse cause and effect (AO3)
- Evaluate pros/cons or short/long-term (AO4), if asked
- Avoid writing a full model answer unless requested

**Structure:**
- ✅ Define
- ✅ Explain cause and effect
- ✅ Apply with real-world example
- ✅ Evaluate (only if “assess”, “evaluate”, or “discuss” is used)

---

### 6. 📊 Diagram Explanation
- Describe diagram elements (axes, lines, equilibrium)
- Explain any shifts or changes
- Link it clearly to the economic concept being tested

**Structure:**
- ✅ Diagram name
- ✅ Axes and labels
- ✅ What it shows
- ✅ Cause of shifts
- ✅ Link to economic outcome

---

### 7. 🧮 Calculation
- Provide correct formula
- Sub in values step-by-step
- Provide the final answer
- Explain what the result means in an economic context

**Structure:**
- ✅ Formula
- ✅ Steps
- ✅ Final answer
- ✅ Interpretation (e.g., “This means demand is elastic”)

---

### 8. 📚 Summary / Revision Notes
- Use **bullet points**
- Be concise, exam-focused, and spec-accurate
- Include:
  - Key definitions
  - Diagrams (described)
  - Formulas
  - Real examples
  - Common exam questions

**Structure:**
- ✅ Subheadings for clarity (e.g. “Price Elasticity of Demand”)
- ✅ Bullet points for revision
- ✅ Tip box if needed: “Remember…”

---

🛑 If the question is unclear or mixes types (e.g., “define and discuss” without clarification), politely ask the user to clarify.

---

📌 Always base answers on the Edexcel A-Level Economics A specification (Themes 1–4). Use appropriate exam terminology, correct command words, and apply real-world examples or UK data if appropriate.

Assessment Objectives to aim for:
- **AO1** – Knowledge and definitions
- **AO2** – Application to real-world or case study
- **AO3** – Logical chains of analysis
- **AO4** – Evaluation and judgement

---

Context:
{context}

Question:
{question}

Answer:
"""
user_prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="Context:\n{context}\n\nQuestion:\n{question}"
)

# === Ask Question Function ===
from difflib import SequenceMatcher

def is_similar(doc_text, query, threshold=0.08):
    """Check if a doc is genuinely related to the query."""
    return SequenceMatcher(None, doc_text.lower(), query.lower()).ratio() > threshold

def ask_question(query, history=None, max_turns=4):
    if history is None or len(history) == 0:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # === Retrieve and filter relevant documents ===
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    relevant_docs = [doc for doc in docs if is_similar(doc.page_content, query)]

    if not relevant_docs:
        user_input = f"No relevant information was found for this question.\n\nQuestion: {query}"
        context = ""
    else:
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        user_input = user_prompt_template.format(question=query, context=context)

    history.append({"role": "user", "content": user_input})

    # === Trim conversation history ===
    max_messages = max_turns * 2
    trimmed = [history[0]] + history[-max_messages:]

    # === Send to Together API ===
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": trimmed,
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 0.9
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('API_KEY')}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        assistant_reply = result["choices"][0]["message"]["content"]

        sources = {doc.metadata.get("source") for doc in relevant_docs if "source" in doc.metadata}
        if sources:
            limited_sources = list(sources)[:3]
            assistant_reply += "\n\n📎 **Sources:**\n" + "\n".join(f"- {src}" for src in limited_sources)

        history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply, history

    except Exception as e:
        return f"⚠️ Error: {e}", history


# === Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📘 A-Level Study Chatbot (Powered by Together AI + Mistral)")

    chat_history = gr.State([])  # stores full history as list of dicts

    chatbot = gr.Chatbot(label="Chat History")
    query = gr.Textbox(label="Ask a question", placeholder="e.g. What’s in Paper 2?")

    answer_btn = gr.Button("Ask")

    def gradio_ask_question(user_query, history):
        # Call your existing ask_question with user_query and full history
        assistant_reply, updated_history = ask_question(user_query, history)

        # Convert updated_history to list of (user_msg, assistant_msg) pairs for display
        pairs = []
        i = 0
        while i < len(updated_history):
            if updated_history[i]["role"] == "user":
                # Show only the raw user query, not the full prompt with context
                user_msg = user_query if i == len(updated_history) - 2 else updated_history[i]["content"]
                assistant_msg = ""
                if i + 1 < len(updated_history) and updated_history[i + 1]["role"] == "assistant":
                    assistant_msg = updated_history[i + 1]["content"]
                pairs.append((user_msg, assistant_msg))
                i += 2
            else:
                i += 1

        return pairs, updated_history, ""  # "" clears the input box

    # Connect both button click and Enter key (Textbox submit)
    query.submit(
        fn=gradio_ask_question,
        inputs=[query, chat_history],
        outputs=[chatbot, chat_history, query]  # query = textbox, will be cleared
    )

    answer_btn.click(
        fn=gradio_ask_question,
        inputs=[query, chat_history],
        outputs=[chatbot, chat_history, query]  # query will be cleared here too
    )

demo.launch()

