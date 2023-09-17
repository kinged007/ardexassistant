import requests
import json
import os
import html2text

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from llama_index import Document
from llama_index import VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st

browserless_api_key = os.getenv("BROWSERLESS_API_KEY")

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
        f"https://chrome.browserless.io/scrape?token={browserless_api_key}",
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



def main():

    st.markdown("<div style='text-align:center;'> <img style='width:340px;' src='https://ardex.co.uk/wp-content/uploads/ardex-logo.png' /></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>ARDEX AI Assistant</h1>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;'>Welcome to ARDEX AI Assistant! Please feel free to ask any question.</p>", unsafe_allow_html=True)

    # Set up memory
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Set up the LLMChain, passing in memory
    
    url = "https://ardexaustralia.com/"
    query = "ARDEX"
    markdown = get_markdown_from_url(url)
    index = create_index_from_text(markdown)
        
    # Get relevant data with similarity search
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    texts = [node.node.text for node in nodes]
    
    docs = ' '.join(texts)
    
    template = "Context: "
    
    template += docs
    
    template += """You are an AI chatbot for ARDEX having a conversation with a human. 
    Please follow the following instructions:
       
    - BEFORE ANSWERING THE QUESTION, ASK A FOLLOW UP QUESTION.
    
    - USE THE CONTEXT PROVIDED TO ANSWER THE USER QUESTION, and PROVIDE ANY IMAGES ASSOCIATED OR RELATED TO THE ANSWER PROVIDED WHENEVER POSSIBLE. DO NOT MAKE ANYTHING UP.
    
    - IF RELEVANT, BREAK YOUR ANSWER DO INTO STEPS
    
    - If asked about any products, show the image of the product asked for, if possible.
    
    - If there are relevant images, video, links, they are very important reference data, please include them as part of the answer
    
    - If suitable to the answer, provide any recommendations to products.
    
    - If suitable to the answer, provide recommendations for training sessions provided by ARDEX.
    
    - FORMAT YOUR ANSWER IN MARKDOWN
    
    - ALWAYS ASK FOLLOW UP QUESTIONS!

    - SHOW RELEVANT IMAGES OF PRODUCTS

    Please consider the follow:
    A common question is what the best way is to tile a swimming pool. Now while this may sound like a straightforward question, there are a lot of considerations depending on the type of application or the type of pool.

     

    Some of the considerations and what the bot ideally should ask are some of the following questions:

     

    What type of pool is it?
    I. e. concrete, fibreglass, shotcrete, etc?
    Is it going to be freshwater or saltwater?
    What type of tile is going to be used?
     

    Depending on the above, it can be any number of products used and different types of applications that need to be considered. Below are some links to the type of documentation that we have:


    {history}
    Human: {human_input}
    AI: """
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    llm_chain = LLMChain(llm=OpenAI(openai_api_key=openai_api_key), prompt=prompt, memory=memory)

    # Render current messages from StreamlitChatMessageHistory
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # If user inputs a new prompt, generate and draw a new response
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        # Note: new messages are saved to history automatically by Langchain during run
        response = llm_chain.run(prompt)
        st.chat_message("ai").write(response)
    
if __name__ == "__main__":
    main()
