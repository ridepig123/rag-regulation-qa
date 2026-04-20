from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(documents: List[Document]) -> List[Document]:
    """
    将 Langchain 文档对象列表进行分块。

    参数:
        documents (List[Document]): 待分块的文档列表。

    返回:
        List[Document]: 分块后的文档列表。
    """
    # 初始化文本分割器
    # RecursiveCharacterTextSplitter 会尝试按 ["\n\n", "\n", " ", ""] 的顺序进行分割
    # 这对于保持段落和句子的完整性很有帮助
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True  # 在metadata中添加块在原文中的起始位置
    )
    
    # 对所有文档进行分块
    split_docs = text_splitter.split_documents(documents)
    
    return split_docs

if __name__ == '__main__':
    # 这是一个示例用法
    
    # 模拟从 document_parser 传入的文档对象
    sample_docs = [
        Document(
            page_content="""这是第一份文档的全部内容。它包含多个句子。
            Langchain的RecursiveCharacterTextSplitter是一个强大的工具。
            它会首先尝试用换行符来分割文本。如果块的大小仍然超过chunk_size，它会继续尝试用其他分隔符。
            """,
            metadata={"source": "sample_doc_1.docx"}
        ),
        Document(
            page_content="这是第二份文档，它比较短。",
            metadata={"source": "sample_doc_2.docx"}
        ),
        Document(
            page_content="| 职级 | 一线城市 | 二线城市 |\n| --- | --- | --- |\n| P1 - P3 | 500 | 400 |\n| P4 - P5 | 700 | 600 |",
            metadata={"source": "sample_doc_1.docx"}
        )
    ]
    
    print("--- 原始文档 ---")
    for doc in sample_docs:
        print(doc)
    
    # 执行分块
    chunked_documents = split_documents(sample_docs)
    
    print(f"\n--- 分块后的文档 (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}) ---")
    print(f"原始文档数量: {len(sample_docs)}, 分块后文档数量: {len(chunked_documents)}")
    
    for i, chunk in enumerate(chunked_documents):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Content:\n{chunk.page_content}")
        print(f"Metadata: {chunk.metadata}")
