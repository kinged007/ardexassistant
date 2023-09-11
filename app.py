from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
#from dotenv import load_dotenv
import requests
import json
import os
import html2text
from langchain.chat_models import ChatOpenAI
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from llama_index import VectorStoreIndex
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
#from embedchain import App
import streamlit as st
import openai

#chatbot = App()

#load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


# 1. Scrape raw HTML

def scrape_website(url: str):

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url,
        "elements": [{
            "selector": "body"
        }]
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(
        f"https://chrome.browserless.io/scrape?token={brwoserless_api_key}",
        headers=headers,
        data=data_json
    )

    # Check the response status code
    if response.status_code == 200:
        # Decode & Load the string as a JSON object
        result = response.content
        data_str = result.decode('utf-8')
        data_dict = json.loads(data_str)

        # Extract the HTML content from the dictionary
        html_string = data_dict['data'][0]['results'][0]['html']

        return html_string
    else:
        print(f"HTTP request failed with status code {response.status_code}")


# 2. Convert html to markdown

def convert_html_to_markdown(html):

    # Create an html2text converter
    converter = html2text.HTML2Text()

    # Configure the converter
    converter.ignore_links = False

    # Convert the HTML to Markdown
    markdown = converter.handle(html)

    return markdown


# Turn https://developers.webflow.com/docs/getting-started-with-apps to https://developers.webflow.com

def get_base_url(url):
    parsed_url = urlparse(url)

    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


# Turn relative url to absolute url in html

def convert_to_absolute_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')

    for img_tag in soup.find_all('img'):
        if img_tag.get('src'):
            src = img_tag.get('src')
            if src and src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['src'] = absolute_url
        elif img_tag and img_tag.get('data-src'):
            src = img_tag.get('data-src')
            if src and src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['data-src'] = absolute_url

    for link_tag in soup.find_all('a'):
        href = link_tag.get('href')
        #print(href)
        if href and href.startswith(('http://', 'https://')):
            continue
        absolute_url = urljoin(base_url, href)
        link_tag['href'] = absolute_url

    updated_html = str(soup)

    return updated_html


def get_markdown_from_url(url):
    base_url = get_base_url(url)
    html = scrape_website(url)
    updated_html = convert_to_absolute_url(html, base_url)
    markdown = convert_html_to_markdown(updated_html)

    return markdown


# 3. Create vector index from markdown

def create_index_from_text(markdown):
    text_splitter = TokenTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=20,
        backup_separators=["\n\n", ".", ","]
    )

    node_parser = SimpleNodeParser(text_splitter=text_splitter)
    nodes = node_parser.get_nodes_from_documents(
        [Document(text=markdown)], show_progress=True)

    # build index
    index = VectorStoreIndex(nodes)

    print("Index created!")
    return index


# 4. Retrieval Augmented Generation (RAG)


def generate_answer(query, index):

    # Get relevant data with similarity search
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    texts = [node.node.text for node in nodes]

    print("Retrieved texts!", texts)




    # Generate answer with OpenAI
    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")
    template = """
    CONTEXT: {docs}
    You are a helpful assistant for ARDEX, respond as human-like as possible, above is some context, 
    Also, you are  an ai chatbot for ARDEX's website. However, to not repond with "Based on the context given,..." or any phrases or sentences. Just leave it out.
    Please answer the question, and make sure you follow ALL of the rules below:
    1. Answer the questions only based on context provided, do not make things up
    2. Answer questions in a helpful manner that straight to the point, with clear structure & all relevant information that might help users answer the question
    3. Anwser should be formatted in Markdown
    4. If asked about any products, show the image of the product asked for, if possible.
    5. If there are relevant images, video, links, they are very important reference data, please include them as part of the answer
    6. If suitable to the answer, provide any recommendations to products.
    7. If suitable to the answer, provide recommendations for training sessions provided by ARDEX.
    8. Place all media such as images on a separate line.
    9. Show the images, DO NOT PROVIDE A LINK TO THE IMAGE.
    Use your judgement, If you think you can provide a more refined answer if you had more information from the USER, please ask any follow up questions to provide a more refined answer to the question

    DO NOT MAKE ANYTHING UP!
    DO NOT RESPOND AS IF YOU ARE GIVEN A CONTEXT, YOU ARE TO RESPOND AS HUMAN LIKE AS POSSIBLE.
    IF YOUR RESPONSE WOULD INCLUDE "Based on the context, provided..". DO NOT SAY. ACT LIKE A HUMAN THAT IS KNOWLEDGEABLE ABOUT EVERYTHING ARDEX.
    
    QUESTION: {query}
    ANSWER (formatted in Markdown):

    """
    cut =  """
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(template)

    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(embedding_function=embeddings, persist_directory=directory)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        model,
        vectordb.as_retriever(),
    	condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    	memory=memory
    )

    print(qa({"question":query}))
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    response = chain.invoke({"docs": texts, "query": query})

    return response.content

# Setup Sttreamlit page layout
st.set_page_config(layout="wide")

st.markdown("<div style='text-align:center;'> <img style='width:340px;' src='https://ardex.co.uk/wp-content/uploads/ardex-logo.png' /></div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>ARDEX AI Assistant</h1>", unsafe_allow_html=True)


def main():
    url = "https://ardexaustralia.com/"
    query = st.text_input("How may I assist you?")
    markdown = get_markdown_from_url(url)
    index = create_index_from_text(markdown)
    answer = generate_answer(query, index)
    st.markdown(answer)

if __name__ == "__main__":
    main()
