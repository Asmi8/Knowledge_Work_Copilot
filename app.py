import streamlit as st
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import tempfile


st.set_page_config(
    page_title="Knowledge Work Copilot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Generative AI Assistant for Knowledge Work")
st.write(
    "Upload a PDF document and use AI to summarize it, extract keywords, "
    "answer questions, and generate a structured analysis report."
)


@st.cache_resource
def load_summarization_model():
    model_name = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return tokenizer, model, device


@st.cache_resource
def load_qa_model():
    qa = pipeline(
        task="question-answering",
        model="distilbert-base-cased-distilled-squad"
    )
    return qa


def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    reader = PdfReader(temp_path)
    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {page_number} ---\n"
            text += page_text + "\n"

    return text


def summarize_chunk(chunk, tokenizer, model, device, max_length=160, min_length=50):
    inputs = tokenizer(
        chunk,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_long_text(text, tokenizer, model, device, chunk_size=2500, max_chunks=5):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 500:
            chunks.append(chunk)

    summaries = []

    total_chunks = min(len(chunks), max_chunks)

    for chunk in chunks[:max_chunks]:
        summary = summarize_chunk(chunk, tokenizer, model, device)
        summaries.append(summary)

    combined_summary = "\n\n".join(summaries)

    if len(combined_summary) > 1000:
        final_summary = summarize_chunk(
            combined_summary,
            tokenizer,
            model,
            device,
            max_length=220,
            min_length=80
        )
        return final_summary

    return combined_summary


def extract_keywords_tfidf(text, top_n=15):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    keyword_scores = list(zip(feature_names, scores))
    keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)

    return keyword_scores[:top_n]


def answer_question(question, document_text, qa_model, context_size=4000):
    context = document_text[:context_size]

    result = qa_model(
        question=question,
        context=context
    )

    return result


def generate_report(summary, keywords, qa_result=None, question=None):
    keyword_list = ", ".join([keyword for keyword, score in keywords])

    report = f"""
GENERATIVE AI ASSISTANT FOR KNOWLEDGE WORK
Document Analysis Report

1. DOCUMENT SUMMARY

{summary}

2. IMPORTANT KEYWORDS

{keyword_list}

3. BUSINESS VALUE

This prototype helps knowledge workers quickly understand long documents by automatically extracting text from PDFs, summarizing important content, identifying major keywords, and answering document-based questions. It reduces manual reading time and supports faster research, reporting, and decision-making.

4. LIMITATIONS

- This free deployed version uses lightweight Hugging Face models for faster inference.
- Summary quality may be lower than commercial LLMs such as Microsoft Copilot, OpenAI, or Gemini.
- The question-answering model answers from a limited text window.
- Very long or scanned PDFs may not extract text perfectly.

5. FUTURE SCOPE

- Improve long-document retrieval using embeddings and vector search.
- Add support for DOCX and TXT files.
- Add email drafting and slide-outline generation.
- Add user authentication and cloud storage.
- Integrate with Microsoft 365, SharePoint, OneDrive, or Microsoft Graph in a future enterprise version.
"""

    if qa_result and question:
        report += f"""

6. SAMPLE QUESTION ANSWERING

Question:
{question}

Answer:
{qa_result["answer"]}

Confidence Score:
{qa_result["score"]:.4f}
"""

    return report


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    with st.spinner("Extracting text from PDF..."):
        document_text = extract_text_from_pdf(uploaded_file)

    st.subheader("Document Preview")
    st.text_area("Extracted Text Preview", document_text[:3000], height=250)

    if len(document_text.strip()) == 0:
        st.error("No text could be extracted. This may be a scanned PDF.")
    else:
        tokenizer, summarization_model, device = load_summarization_model()
        qa_model = load_qa_model()

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Summary", "Keywords", "Question Answering", "Report"]
        )

        with tab1:
            st.subheader("Document Summary")

            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = summarize_long_text(
                        document_text,
                        tokenizer,
                        summarization_model,
                        device
                    )

                st.session_state["summary"] = summary
                st.write(summary)

        with tab2:
            st.subheader("Important Keywords")

            if st.button("Extract Keywords"):
                keywords = extract_keywords_tfidf(document_text)
                st.session_state["keywords"] = keywords

                for keyword, score in keywords:
                    st.write(f"- **{keyword}**: {score:.4f}")

        with tab3:
            st.subheader("Ask a Question About the Document")

            question = st.text_input(
                "Example: What is the main topic of this document?"
            )

            if st.button("Get Answer"):
                if question.strip() == "":
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Searching for answer..."):
                        qa_result = answer_question(
                            question,
                            document_text,
                            qa_model
                        )

                    st.session_state["question"] = question
                    st.session_state["qa_result"] = qa_result

                    st.write("**Answer:**")
                    st.write(qa_result["answer"])

                    st.write("**Confidence Score:**")
                    st.write(round(qa_result["score"], 4))

        with tab4:
            st.subheader("Generate Analysis Report")

            if st.button("Generate Report"):
                if "summary" not in st.session_state:
                    with st.spinner("Generating summary first..."):
                        st.session_state["summary"] = summarize_long_text(
                            document_text,
                            tokenizer,
                            summarization_model,
                            device
                        )

                if "keywords" not in st.session_state:
                    st.session_state["keywords"] = extract_keywords_tfidf(
                        document_text
                    )

                report = generate_report(
                    st.session_state["summary"],
                    st.session_state["keywords"],
                    st.session_state.get("qa_result"),
                    st.session_state.get("question")
                )

                st.session_state["report"] = report

                st.text_area("Generated Report", report, height=500)

                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="knowledge_work_copilot_report.txt",
                    mime="text/plain"
                )

else:
    st.info("Upload a PDF file to begin.")
