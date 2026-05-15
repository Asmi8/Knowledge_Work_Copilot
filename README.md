# Knowledge_Work_Copilot
# Generative AI Assistant for Knowledge Work

## Project Overview

This project is a deployable AI assistant inspired by Microsoft Copilot and generative AI tools for knowledge workers. The application allows users to upload a PDF document, extract text, generate summaries, identify important keywords, ask document-based questions, and download a structured analysis report.

## Problem Statement

Knowledge workers spend significant time reading long documents, extracting insights, preparing summaries, and answering document-specific questions. This project demonstrates how AI and natural language processing can reduce manual effort and support faster research, reporting, and decision-making.

## Features

- PDF upload
- Text extraction from PDF
- AI-based document summarization
- Keyword extraction using TF-IDF
- Document question answering
- Structured report generation
- Downloadable report file

## Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- DistilBART summarization model
- DistilBERT question-answering model
- pypdf
- scikit-learn TF-IDF

## Model Details

The Google Colab prototype used `facebook/bart-large-cnn`, a BART-based transformer model, for summarization.

The deployed Streamlit version uses `sshleifer/distilbart-cnn-12-6`, a lighter DistilBART model, for better performance on free hosting.

For question answering, the deployed version uses `distilbert-base-cased-distilled-squad`.

## System Architecture

User uploads PDF  
→ Text is extracted using pypdf  
→ Summarization model generates document summary  
→ TF-IDF extracts important keywords  
→ QA model answers user questions from document context  
→ System generates downloadable report  

## Limitations

- The deployed version uses lightweight free models, so output quality may be lower than commercial LLMs.
- Very long documents are summarized in chunks.
- The question-answering model uses a limited context window.
- Scanned PDFs may not extract text correctly.
- No Microsoft 365 integration is included in this prototype.

## Future Scope

- Add DOCX and TXT upload support
- Add Retrieval-Augmented Generation
- Improve long-document search using embeddings
- Add email drafting and slide outline generation
- Add Streamlit authentication
- Integrate with Microsoft 365, SharePoint, OneDrive, or Microsoft Graph
