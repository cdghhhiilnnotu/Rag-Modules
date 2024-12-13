from langchain_experimental.text_splitter import SemanticChunker
import pickle

from embeddings import HFEmbedding

class BaseChungking():

    def __init__(self):
        pass

class DocChunk(BaseChungking):

    def __init__(self, embeding_model):
        super().__init__()
        self.core = SemanticChunker(embeding_model, breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90)

    def chunking(self, documents):
        return self.core.split_documents(documents)
    
if __name__ == "__main__":
    # Load documents from pickle file
    with open('sources/documents/all_docs.pkl', "rb") as file:
        loaded_documents = pickle.load(file)

    embedding = HFEmbedding()
    chunker = DocChunk(embeding_model=embedding)
    splitted = chunker.chunking(loaded_documents)

    print(splitted[0])
