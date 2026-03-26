from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_PATH = PROJECT_ROOT / "rag_source"


def load_documents() -> List[Any]:
    docs: List[Any] = []

    if not RAG_PATH.exists():
        return docs

    for file in RAG_PATH.glob("**/*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())

    for file in RAG_PATH.glob("**/*.md"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())

    return docs


@dataclass
class SimpleLocalRAG:
    vectorstore: Any

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        query = inputs.get("query", "").strip()

        if not query:
            return {
                "result": "Please provide a question.",
                "source_documents": [],
            }

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        source_documents = retriever.invoke(query)

        if not source_documents:
            return {
                "result": "I could not find relevant project information in the local knowledge base.",
                "source_documents": [],
            }

        context_blocks = []
        for i, doc in enumerate(source_documents, start=1):
            text = doc.page_content.strip().replace("\n", " ")
            context_blocks.append(f"[Source {i}] {text}")

        context_text = "\n\n".join(context_blocks)

        answer = (
            "Based on the local project knowledge base, here is the most relevant information:\n\n"
            f"{context_text}\n\n"
            "Summary:\n"
            "This project is framed as a non-diagnostic mental health triage system. "
            "It is intended to support early signal detection from text, prioritize potentially higher-risk cases, "
            "and assist human review rather than replace clinical judgment."
        )

        return {
            "result": answer,
            "source_documents": source_documents,
        }


def build_qa_chain() -> Optional[SimpleLocalRAG]:
    docs = load_documents()

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return SimpleLocalRAG(vectorstore=vectorstore)