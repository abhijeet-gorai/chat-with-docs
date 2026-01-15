from services.vector_db.chroma import ChromaRetriever
from services.groq import init_chat_model
from utils.file_readers import read_pdf_file, read_docx_file, read_txt_file
from typing import List, Tuple, Iterator, Union, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, BaseMessageChunk
from prompts.rag_prompts import qa_system_prompt, qa_user_prompt
from utils.rerankers.cross_encoder import CrossEncoderReranker
from utils.rerankers.colbert import ColbertReranker
from langchain_core.documents import Document
from .config_schema import AppConfig

class RerankerRAG:
    """
    A Retrieval-Augmented Generation (RAG) pipeline that leverages a reranking mechanism to improve document relevance.
    
    This implementation retrieves candidate documents from a vector database, uses a reranker to select the top
    relevant documents, and then generates an answer using a language model. It supports both complete responses and streaming.
    """

    def __init__(self, config:AppConfig = {}) -> None:
        """
        Initializes the RerankerRAG system by setting up the language model, vector database, and reranker.
        
        Args:
            config (AppConfig): Application configuration containing keys for LLMs, vector database, 
                                file reader options, reranker and chunking parameters. Defaults to {}.
        """
        self.llm_config = config.get("llm", {})
        vector_db_config = config.get("vector_db", {})
        reranker_config = config.get("reranker", {})
        reranker_type = reranker_config.get("type", "colbert")
        reranker_params = reranker_config.get("params", {})
        self.file_reader = config.get("file_reader", {
            "pdf": {"pdfloader": "PYMUPDF"},
            "docx": {"docxloader": "python-docx"}
        })
        self.chunking_config = config.get("chunking", {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "separators": ["\n\n", " "]
        })
        
        self.llm = init_chat_model(**self.llm_config)
        self.vector_db = ChromaRetriever(**vector_db_config)
        if reranker_type == "colbert":
            self.reranker = ColbertReranker(**reranker_params)
        else:
            self.reranker = CrossEncoderReranker(**reranker_params)

    def set_llm_params(
        self,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> None:
        """
        Update the LLM generation parameters. Only recreates the LLM if any param changed.
        """
        updated = False
        for key, val in {
            'model_id': model_id,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop_sequences': stop_sequences,
        }.items():
            if val is not None and self.llm_config.get(key) != val:
                self.llm_config[key] = val
                updated = True
        if updated:
            self.llm = init_chat_model(**self.llm_config)

    def add_documents(self, file_list: List[str], collection_name: Optional[str] = None) -> None:
        """
        Reads and processes documents from the provided file paths, splits them into chunks, and adds them to the vector database.
        
        Args:
            file_list (List[str]): A list of file paths to be added.
            collection_name (Optional[str]): The collection to add documents to. Defaults to current collection.
        
        Raises:
            ValueError: If a file format is unsupported.
        """
        docs = []
        for file in file_list:
            if file.lower().endswith('.pdf'):
                docs.extend(read_pdf_file(file, **self.file_reader.get("pdf")))
            elif file.lower().endswith('.docx'):
                docs.extend(read_docx_file(file, **self.file_reader.get("docx")))
            elif file.lower().endswith('.txt'):
                docs.extend(read_txt_file(file))
            else:
                raise ValueError(f"Unsupported file format: {file}")
        
        splitter = RecursiveCharacterTextSplitter(**self.chunking_config)
        split_docs = splitter.split_documents(docs)
        
        # Deduplicate documents based on page_content
        seen_contents = set()
        unique_docs = []
        for doc in split_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
        
        self.vector_db.add_documents(unique_docs, collection_name=collection_name)

    def get_context(self, query: str, k: int = 4, collection_name: Optional[str] = None) -> List[Document]:
        """
        Retrieves and reranks relevant documents based on the query.
        
        The method first retrieves a larger set of candidate documents (k*3) from the vector database, then uses a reranker
        to select the top k documents.
        
        Args:
            query (str): The user query.
            k (int, optional): The number of top relevant documents to return. Defaults to 4.
            collection_name (Optional[str]): The collection to retrieve from. Defaults to current collection.
        
        Returns:
            List[Document]: A list of the top k reranked documents relevant to the query.
        """
        # Retrieve more documents than needed for reranking.
        candidate_docs = self.vector_db.get_relevant_docs(query=query, k=k * 3, collection_name=collection_name)
        # Rerank the candidate documents and select the top k.
        reranked_docs = self.reranker.rerank(query, candidate_docs)[:k]
        return reranked_docs

    def get_answer(
        self, 
        query: str, 
        docs: List[Document], 
        stream: bool = False
    ) -> Union[BaseMessage, Iterator[BaseMessageChunk]]:
        """
        Generates an answer to the query based on the provided context from retrieved documents.
        
        The context is formed by concatenating the page content of the documents, which is then provided to the language model
        along with system and user prompts.
        
        Args:
            query (str): The user query.
            docs (List[Document]): The list of documents used to form the context.
            stream (bool, optional): Whether to stream the answer. Defaults to False.
        
        Returns:
            Union[BaseMessage, Iterator[BaseMessageChunk]]:
                - If stream is False, returns a single BaseMessage containing the complete answer.
                - If stream is True, returns an iterator over BaseMessageChunk instances.
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = [
            SystemMessage(content=qa_system_prompt),
            HumanMessage(content=qa_user_prompt.format(query=query, context=context))
        ]
        if stream:
            response = self.llm.stream(messages)
        else:
            response = self.llm.invoke(messages)
        return response

    def respond_to_query(self, query: str, k: int = 4, collection_name: Optional[str] = None) -> Tuple[BaseMessage, List[Document]]:
        """
        Processes the query by retrieving relevant documents, generating a complete answer, and returning both.
        
        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve. Defaults to 4.
            collection_name (Optional[str]): The collection to retrieve from. Defaults to current collection.
        
        Returns:
            Tuple[BaseMessage, List[Document]]:
                A tuple containing:
                - The generated answer as a BaseMessage.
                - The list of relevant documents used to generate the answer.
        """
        similar_docs = self.get_context(query=query, k=k, collection_name=collection_name)
        answer = self.get_answer(query, similar_docs)
        return answer, similar_docs

    def stream_answer(self, query: str, k: int = 4, collection_name: Optional[str] = None) -> Tuple[Iterator[BaseMessageChunk], List[Document]]:
        """
        Processes the query by retrieving relevant documents and streaming the answer.
        
        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve. Defaults to 4.
            collection_name (Optional[str]): The collection to retrieve from. Defaults to current collection.
        
        Returns:
            Tuple[Iterator[BaseMessageChunk], List[Document]]: A tuple containing an iterator over the 
                generated answer chunks and the list of relevant documents.
        """
        similar_docs = self.get_context(query=query, k=k, collection_name=collection_name)
        answer = self.get_answer(query, similar_docs, stream=True)
        return answer, similar_docs
