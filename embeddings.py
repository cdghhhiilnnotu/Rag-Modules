from langchain_community.embeddings import HuggingFaceEmbeddings

class BaseEmbedding():

    def __init__(self):
        pass

class HFEmbedding(BaseEmbedding):

    def __init__(self, model_name='keepitreal/vietnamese-sbert'):
        super().__init__()
        self.core = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, documents):
        return self.core.embed_documents(documents)