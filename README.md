# Chat_with_PDF
A Chatbot application of RAG(Retrieval Augmented Generation) that reads and evaluates the text in a PDF file and answers questions regarding that text.

### Tired of reading your whole documents with loads of irrelevant information that you don't need? Or you want to scrape a book for one single thing, but don't want to read the whole book? or You are looking for an understanding of a concept in a different language that you don't speak or understand. Don't Worry! this project has you covered with an additional layer of privacy to run it locally on your pc without the need for internet or a **"PAID API"**. It has the same interface like ChatGPT or other chatbots. You can ask your personal assistant to do analysis of a data you are working on, or make you understand a concept that will take too long to understand via the conventional method. The better your prompts are, the better this assistant will work for you.

## Example 1
A self help-book was given to the LLM and it gave the following output.

![Output](/assets/output.png)

## Example 2
Another example from the same text.

![Output2](/assets/output2.png)

## Setup
To setup this repository, you need to have anaconda installed on your system.

Clone this repository in your system:
```bash
git clone https://github.com/mtayyab2/Chat_with_PDF.git
cd Chat_with_PDF
conda create -m Chat_with_PDF python=3.8
conda activate Chat_with_PDF
```

To install the required libraries:
```bash
pip install numpy sentence-transformers pypdf2 streamlit ollama
```
To run it locally:
```bash
streamlit run chat_with_pdf.py
```

## Overview

This app takes a PDF file and look for its embeddings in the embeddings folder, if the embeddings are not available it creates a json file with textual embeddings using a Huggingface SentenceTransformer library `'all-MiniLM-L6-v2'`. You can select the LLM model you want to use to chat with the PDF (*different models will have different outputs for same prompts*).

The emebeddings for the given prompt is created and then a similarity check is done to get the relevant information from the given context(*pdf*). A `numpy` normality function is used to check for similarity between embeddings.

#### NOTE: The first a file is loaded, the system will generate an embeddings file which can take a lot of time depending on the system you are using. Once an embedding file has been created, you can use it again without the wait. Embeddings for the prompts are generated at runtime which usually take about 2-10 seconds.

## Contribution

Feel free to contribute to this repository by forking or if you find any issue, raise an issue in the ISSUES tab.