# 🕵️ Fake News Detection System with RAG using Endee

An AI-powered Fake News Detection system that combines **Machine Learning Classification** with **Retrieval-Augmented Generation (RAG)**. This project uses the **Endee Vector Database** for efficient semantic search and evidence retrieval.

## 🚀 Project Overview
This application helps users verify news articles. It performs two main actions:
1.  **Classification:** Uses a Logistic Regression model to predict if a news piece is "REAL" or "FAKE" based on linguistic patterns.
2.  **Retrieval (RAG):** Uses the **Endee Vector Database** to find the top 3 most similar articles from a dataset of 70,000+ records to provide context/evidence to the user.

## 🛠️ System Design
The system architecture consists of three main components:
* **Data Processor:** Cleans and standardizes the `news.csv` dataset.
* **ML Engine:** A Scikit-Learn pipeline (TF-IDF + Logistic Regression) for real-time classification.
* **Vector Engine (Endee):** Stores 1536-dimensional embeddings (using `all-MiniLM-L6-v2`) in the **Endee Vector DB** for high-speed semantic retrieval.



## 🏗️ Use of Endee
Endee is used as the core Vector Database to enable the RAG workflow. 
- **Storage:** Metadata (original text and labels) is stored alongside vectors.
- **Search:** When a user inputs a news story, we generate an embedding and perform a similarity search in Endee to find matching historical records.

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Git

### Installation
1. **Fork and Clone** the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/endee.git](https://github.com/YOUR_USERNAME/endee.git)
   cd endee