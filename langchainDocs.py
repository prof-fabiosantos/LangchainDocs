# Import libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

# Create a new openai api key
os.environ["OPENAI_API_KEY"] = ""
# set up openai api key
openai_api_key = os.environ.get('OPENAI_API_KEY')

#from langchain.document_loaders import PyPDFLoader
#loader = PyPDFLoader("documentos/TCC Erica Amoedo.pdf")
#doc = loader.load()

from langchain.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader("documentos/")
doc = loader.load()

# Print number of txt files in directory
#loader = DirectoryLoader('documentos', glob="./*.txt")
#doc = loader.load ( )
#print(len(doc))

# Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(doc)

# Count the number of chunks
print(len(texts))

print(texts[0])

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

# OpenAI embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# Persist the db to disk
vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory,
                   embedding_function=embedding)


# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

#docs = retriever.get_relevant_documents("What to do when getting started?")
#print(len(docs))
#print(retriever.search_type)

"""
# Create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True)

"""

# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

# Create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True)
# Cite sources
"""
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
"""

# Cite sources
def process_llm_response(llm_response):
    result = llm_response['result']
    print(result)
    
    print('\n\nSources:')
    sources = set()  # Use a set to avoid duplicate sources
    for source in llm_response["source_documents"]:
        sources.add(source.metadata['source'])
    for source in sources:
        print(source)
    
query = "Quem é Erica Amoedo?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

from chromaviz import visualize_collection
visualize_collection(vectordb._collection)
