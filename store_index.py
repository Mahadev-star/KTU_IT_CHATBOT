from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY= os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

#print(PINECONE_API_ENV)
#print(PINECONE_API_KEY)

extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()


from langchain_pinecone import PineconeVectorStore
docsearch=PineconeVectorStore.from_documents(documents=text_chunks,
                                             index_name="ai",
                                             embedding=embeddings)