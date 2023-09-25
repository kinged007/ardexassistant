import os
import glob
import streamlit as st

from web_scraper import get_markdown_from_url, create_index_from_text

from PyPDF2 import PdfReader

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate

openai_api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Main function
def main():

    # Set up Streamlit interface
        
    st.markdown("<div style='text-align:center;'> <img style='width:340px;' src='https://ardex.co.uk/wp-content/uploads/ardex-logo.png' /></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>ARDEX AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Welcome to ARDEX AI Assistant! Please feel free to ask any question.</p>", unsafe_allow_html=True)

        # Set up memory
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)

    # Check if there are no previous chat messages
    if len(msgs.messages) == 0:
        # Display initial message only at the very start
        st.chat_message("ai").write("How can I help you?")  # AI's initial message

    # Initialize query_input as None
    query_input = None

    # Get user input for query
    query_input = st.chat_input("Your message")  # Unique key for query input
            
    # Display user's query in the interface if query_input is set
    
    if query_input:
        st.chat_message("human").write(query_input)
        
        # Process PDF documents
        docs_directory = os.path.join(os.getcwd(), 'docs')  # Use absolute path to 'docs' directory
        pdf_files = glob.glob(os.path.join(docs_directory, '*.pdf'))

        if "docs_processed" not in st.session_state:
            st.session_state.docs_processed = False
            
        if not st.session_state.docs_processed:
            for pdf_file in pdf_files:
                with open(pdf_file, 'rb') as file:
                    # Perform processing on each PDF file
                    raw_text = get_pdf_text([file])

                    st.session_state.pdf_text = ''.join(raw_text)
                    st.session_state.docs_processed = True 
        
        url = "https://ardexaustralia.com/"

        # Scrap website url and retrieve markdown
        markdown = get_markdown_from_url(url)

        # Combine pdf_text and markdown
        all_info = st.session_state.pdf_text + markdown

        index = create_index_from_text(all_info)

        # Get relevant data with similarity search
        retriever = index.as_retriever()

        print(f'Query: {query_input}')

        # Input query into the retriever
        nodes = retriever.retrieve(query_input)

        texts = [node.node.text for node in nodes]

        st.session_state.docs = ' '.join(texts)

        # Check if a query is provided before generating a prompt
        if query_input:
            template = "Context: "
            template += st.session_state.docs
            template += """You are an AI staff training assistant for ARDEX having a conversation with a human. Act as a trainer, so you should response in a directional like manner. Your purpose is to train staff members on ARDEX and the products they offer so they may relay the correct information. Before responding, ask the most appropriate follow up question to provide the best possible solution for their query.
            

            Please follow the following instructions:
             
            - Assist the staff on how to use the ARDEX products and tell the user what the product is best for. As a staff member, it is important to understand the products and the systems that ARDEX offers. ARDEX has a wide range of products that are designed for any tiling, flooring or waterproofing application. We provide solutions for subfloor preparation, tile and natural stone installations, tile grouts, waterproofing and roofing membranes, and more. Additionally, ARDEX has an extensive suite of resources to help you. We have the ARDEX System Selector, which is a simple and easy to use solution that helps you find the right system for any job. We also have the Architectural Hub, which provides CAD detail drawings, CPD training, specification sheets, system specifications, and more. Please help them do so with your responses to their queries
             
            - Make appropriate suggestions on ARDEX products to use for the any relatable query the user may make.
               
            - BEFORE ANSWERING THE QUESTION, ASK A FOLLOW UP QUESTION.
            
            - USE THE CONTEXT PROVIDED TO ANSWER THE USER QUESTION. DO NOT MAKE ANYTHING UP.
            
            - IF RELEVANT, BREAK YOUR ANSWER DO INTO STEPS
            
            - If suitable to the answer, provide any recommendations to products.
            
            - FORMAT YOUR ANSWER IN MARKDOWN
            
            - ALWAYS ASK FOLLOW UP QUESTIONS!

            - SHOW RELEVANT IMAGES OF PRODUCTS

            {history}
            Human: {human_input}
            AI: """

            prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
            llm_chain = LLMChain(llm=OpenAI(openai_api_key=openai_api_key), prompt=prompt, memory=memory)

            # Render current messages from StreamlitChatMessageHistory
            for msg in msgs.messages:
                st.chat_message(msg.type).write(msg.content)

            # If user inputs a new prompt, generate and draw a new response
            if query_input:
                
                # Note: new messages are saved to history automatically by Langchain during run
                response = llm_chain.run(query_input)

                print(f'Response: {response}')
    
                # Hide Spinner
                
                st.chat_message('ai').write(response)

if __name__ == '__main__':
    main()
