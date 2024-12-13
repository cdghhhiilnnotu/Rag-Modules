from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os

from embeddings import HFEmbedding
from chunking import DocChunk
import pickle

class VectoreStore():

    db_version = len(os.listdir('./sources/database'))

    def __init__(self, db_path=f'./sources/database/v{db_version}', model_name='keepitreal/vietnamese-sbert'):
        self.embeddings = HFEmbedding()
        self.db_path = db_path

    def create_db(self, chunks: list[str]):
        sentences = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(sentences)
        embeddings_array = np.array(embeddings)
        print(embeddings_array.shape)

        d = embeddings_array.shape[1]

        index = faiss.IndexFlatL2(d)
        index.add(embeddings_array)

        docstore = InMemoryDocstore(
            {i: doc for i, doc in enumerate(chunks)}
        )

        index_to_docstore_id = {i: i for i in range(len(chunks))}

        db = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        db.save_local(self.db_path)
        return db
    
    def get_db(self):
        db = FAISS.load_local(self.db_path, embeddings=self.embeddings)
        return db
    

if __name__ == "__main__":
    # Load documents from pickle file
    with open('sources/documents/all_docs.pkl', "rb") as file:
        loaded_documents = pickle.load(file)

    embedding = HFEmbedding()
    vdb = VectoreStore()
    chunker = DocChunk(embeding_model=embedding)
    chunks = chunker.chunking(loaded_documents)

    db = vdb.create_db(chunks=chunks)

    question = "Học phần là gì?"

    retrieved_docs = db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(question)

    print(retrieved_docs)

    db1 = vdb.get_db()

    question = "Học phần là gì?"

    retrieved_docs = db1.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(question)

    print(retrieved_docs)



