# **Chat with Document**

This project is designed to [brief project description]. This README provides instructions to create a Docker image and run it.

## **Prerequisites**

Before you begin, ensure you have met the following requirements:

- You have installed Docker. You can download it from [Docker's official website](https://www.docker.com/get-started).
- You have a terminal or command prompt access.

## **Building the Docker Image**

Follow these steps to build the Docker image:

**1.** **Create app.py in your current directory.**

The `app.py` file provided is a Streamlit application that enables users to upload multiple PDF documents, process them, and interact with the content via a chatbot interface.

```python
import streamlit as st
from dotenv import load_dotenv
import os
openai_api_key = os.getenv('OPENAI_API_KEY')
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# This method takes a list of PDF files as input, reads them using PdfReader from PyPDF2, 
# and concatenates the text content from all pages into a single string
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# This method splits the provided text into smaller chunks. It uses the CharacterTextSplitter from langchain library
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# This method converts text chunks into embeddings and creates a FAISS vector store.
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    st.success("Pdf Loaded Sucessfully")
    return vectorstore

# This method initializes a conversational retrieval chain using a language model and a vector store
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# This method handles user input by passing the question to the conversation chain and displaying the response
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

#  This is the main function which orchestrates the entire process. It sets up 
# the Streamlit app, handles user input, and processes PDF documents.
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
```

**2. Create a `htmlTemplates.py` in your current directory.**

This Html template is used render the chat theme.

```python
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
```

**3. Create `.env` in your current directory.**

Add OpenAI api key in environment file

```python
OPENAI_API_KEY=xxxxxxxxxx
```

**4. Create `requirements.txt` in your current directory.**

The Librarires required to run the streamlit application

```python
langchain
langchain-community
langchain-core
langchain_openai
PyPDF2
python-dotenv
streamlit
openai
faiss-cpu
altair
tiktoken
```

**5. Create a `Dockerfile` in your current directory.**

```dockerfile
# Use the official lightweight Python image based on Alpine Linux
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8081

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8081", "--server.address", "0.0.0.0"]
```

**6. Build the Docker image with a tag (name) of your choice:**

Open terminal in you current directory and run the below command , make sure that your docker desktop is running.

```bash
docker build -t chat_with_document .
```

## **Running the Docker Container**

Once you have built the Docker image, you can run it as a container:

**1. Run the Docker container in the background:**

```bash
docker run -d -p 8081:8081 chat_with_document
```

This command maps port 8081 on your host to port 8081 on the container. Adjust the ports as necessary.

**2. Verify the container is running:**

```bash
docker ps -a
```

This command lists all running containers. You should see your container listed.

## **Accessing the Application**

You can now access the application in your web browser at `http://localhost:8081`.

## **Stopping the Container**

To stop the running container, use the `docker stop` command followed by the container ID or name:

```bash
docker stop container-id
```
