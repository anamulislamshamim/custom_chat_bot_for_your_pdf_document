import os 
import torch 
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFaceHub 
from ibm_watson_machine_learning.foundation_models.extentions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watson_machine_learning.foundation_model import Model 

from langchain import Prompttemplate  
from langchain.chains import LLMChain, SimpleSequentialChain 

# check for GPU availability and set the appropriate device for computation 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cup" 

# Gloval variables 
conversation_retrieval_chain = None 
chat_history = []
llm_hub = None 
embeddings = None 

# Function to initialize the language model and its embeddings 
def init_llm():
    global llm_hub, embeddings
    
    my_credentials = {
        "url": "https://us-south.ml.cloud.ibm.com", 
        # "key": "secret"
    }
    
    params = {
        GenParams.MAX_NEW_TOKENS: 256, # specify the answer size 256 words
        GenParams.TEMPERATURE: 0.1, # Reduce the randomness
    }
    
    LLAMA2_model = Model(
        model_id = 'meta-llama/llama-2-70b-chat',
        credentials = my_credentials,
        params = params,
        project_id = "skills-network", 
    )
    
    # Initialize the language model 
    llm_hub = WatsonxLLM(model=LLAMA2_model) 
    
    # Initialize embeddings using a pre-trained model to represent the text data
    embeddings = HuggingFaceInstructEmbeddings(
        model_name = "sentence-transformers/all-MiniLL-L6-v2", 
        model_kwargs = {"device": DEVICE}
    )


# Function to process a PDF document 
def process_pdf_document(pdf_document_path):
    global conversation_retrieval_chain
    
    # load the pdf document 
    loader = PyPDFLoader(pdf_document_path) 
    documents = loader.load()
    
    # Split the document into small chunks(chunk_size=1024) and chunk_overlap = 64 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_document(documents) 
    
    # create an embeddings database using chroma from the split text chunks 
    db = Chroma.from_document(texts, embeddings=embeddings)
    
    # Build the QA chain, which utilizes the LLM and retriever for answering questions
    # By default, vectorestore retriever uses similar search.
    # If the underlying vectorstore support maximum marginal relevance search, 
    # you can specify that as the search type (search_type="mmr")
    # You can also specify search kwargs like k to use when doing retrieval. K represent 
    # how many search results send to llm 
    
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key = "question"
        # chain_type_kwargs={"prompt": prompt} # if you are using prompt template, you need to uncomment this part
    )
    
    
# Function to process a user prompt 
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    
    # Query the model 
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history}) 
    answer = output["result"]
    
    # update the chat history
    chat_history.append((prompt, answer))
    
    # Return the model\'s response 
    return answer  