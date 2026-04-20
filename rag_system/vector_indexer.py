"""
向量化与索引模块

负责将文档块转换为向量并构建FAISS索引，支持本地和在线两种嵌入模型。
"""

from typing import List, Optional, Dict, Any
import os
import pickle
import numpy as np
import faiss
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import requests
import json

from .config import (
    SILICONFLOW_API_KEY,
    LOCAL_EMBEDDING_MODEL,
    ONLINE_EMBEDDING_MODEL,
    SILICONFLOW_BASE_URL,
    FAISS_INDEX_PATH_LOCAL,
    FAISS_INDEX_PATH_ONLINE,
    BASE_DIR
)


class EmbeddingModel:
    """嵌入模型基类"""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为向量"""
        raise NotImplementedError


class LocalEmbeddingModel(EmbeddingModel):
    """本地嵌入模型（基于sentence-transformers）"""
    
    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        """
        初始化本地嵌入模型
        
        Args:
            model_name: 模型名称
        """
        print(f"正在加载本地嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name, local_files_only=True, trust_remote_code=False)
        print("本地嵌入模型加载完成")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量数组，形状为 (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings


class OnlineEmbeddingModel(EmbeddingModel):
    """在线嵌入模型（基于SiliconFlow API）"""
    
    def __init__(self, api_key: str = SILICONFLOW_API_KEY, model_name: str = ONLINE_EMBEDDING_MODEL):
        """
        初始化在线嵌入模型
        
        Args:
            api_key: API密钥
            model_name: 模型名称
        """
        if not api_key:
            raise ValueError("API密钥不能为空，请设置SILICONFLOW_API_KEY环境变量")
        
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = SILICONFLOW_BASE_URL
        print(f"初始化在线嵌入模型: {model_name}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量数组，形状为 (len(texts), embedding_dim)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # SiliconFlow API 支持批量嵌入
        data = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = []
            
            # 按照输入顺序提取嵌入向量
            for item in sorted(result['data'], key=lambda x: x['index']):
                embeddings.append(item['embedding'])
            
            return np.array(embeddings, dtype=np.float32)
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"调用在线嵌入API失败: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"解析在线嵌入API响应失败: {e}")
        except Exception as e:
            raise RuntimeError(f"调用在线嵌入API失败: {e}")


class VectorIndexer:
    """向量索引器"""
    
    def __init__(self, use_online: bool = False):
        """
        初始化向量索引器
        
        Args:
            use_online: 是否使用在线嵌入模型
        """
        self.use_online = use_online
        
        # 初始化嵌入模型
        if use_online:
            if not SILICONFLOW_API_KEY:
                raise ValueError("使用在线模型需要设置SILICONFLOW_API_KEY")
            self.embedding_model = OnlineEmbeddingModel()
            self.index_path = FAISS_INDEX_PATH_ONLINE
        else:
            self.embedding_model = LocalEmbeddingModel()
            self.index_path = FAISS_INDEX_PATH_LOCAL
        
        # FAISS索引和文档存储
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.document_metadata: List[Dict[str, Any]] = []
    
    def build_index(self, documents: List[Document]) -> None:
        """
        构建FAISS索引
        
        Args:
            documents: 文档列表
        """
        if not documents:
            raise ValueError("文档列表不能为空")
        
        print(f"开始构建索引，共有 {len(documents)} 个文档块")
        
        # 提取文本内容
        texts = [doc.page_content for doc in documents]
        
        # 向量化
        print("正在进行向量化...")
        embeddings = self.embedding_model.encode(texts)
        
        # 构建FAISS索引
        embedding_dim = embeddings.shape[1]
        print(f"向量维度: {embedding_dim}")
        
        # 使用IndexFlatIP (内积) 进行相似度计算
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # 标准化向量以使用余弦相似度
        faiss.normalize_L2(embeddings)
        
        # 添加向量到索引
        self.index.add(embeddings.astype(np.float32))
        
        # 存储文档和元数据
        self.documents = documents.copy()
        self.document_metadata = [doc.metadata for doc in documents]
        
        print(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")
    
    def save_index(self) -> None:
        """保存索引到磁盘"""
        if self.index is None:
            raise ValueError("索引尚未构建，无法保存")
        
        # 确保目录存在
        index_dir = os.path.dirname(self.index_path)
        os.makedirs(index_dir, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        
        # 保存文档和元数据
        with open(f"{self.index_path}_docs.pkl", 'wb') as f:
            pickle.dump({
                'documents': [{'page_content': doc.page_content, 'metadata': doc.metadata} 
                            for doc in self.documents],
                'metadata': self.document_metadata
            }, f)
        
        mode_name = "在线" if self.use_online else "本地"
        print(f"{mode_name}模式索引已保存到: {self.index_path}")
    
    def load_index(self) -> bool:
        """
        从磁盘加载索引
        
        Returns:
            加载是否成功
        """
        try:
            # 加载FAISS索引
            if not os.path.exists(f"{self.index_path}.faiss"):
                return False
            
            self.index = faiss.read_index(f"{self.index_path}.faiss")
            
            # 加载文档和元数据
            with open(f"{self.index_path}_docs.pkl", 'rb') as f:
                data = pickle.load(f)
                
            # 重建Document对象
            self.documents = [Document(page_content=doc['page_content'], metadata=doc['metadata']) 
                            for doc in data['documents']]
            self.document_metadata = data['metadata']
            
            mode_name = "在线" if self.use_online else "本地"
            print(f"{mode_name}模式索引加载成功，包含 {self.index.ntotal} 个向量")
            return True
            
        except Exception as e:
            print(f"索引加载失败: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索最相似的文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            搜索结果列表，每个元素包含document、score和metadata
        """
        if self.index is None:
            raise ValueError("索引尚未构建或加载")
        
        # 向量化查询
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # 组装结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # 确保索引有效
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'metadata': self.document_metadata[idx],
                    'rank': i + 1
                })
        
        return results


def create_vector_indexer(use_online: bool = False) -> VectorIndexer:
    """
    创建向量索引器的工厂函数
    
    Args:
        use_online: 是否使用在线模型
        
    Returns:
        VectorIndexer实例
    """
    return VectorIndexer(use_online=use_online)


def build_and_save_index(documents: List[Document], use_online: bool = False) -> VectorIndexer:
    """
    构建并保存索引的便捷函数
    
    Args:
        documents: 文档列表
        use_online: 是否使用在线模型
        
    Returns:
        构建好的VectorIndexer实例
    """
    indexer = create_vector_indexer(use_online=use_online)
    indexer.build_index(documents)
    indexer.save_index()
    return indexer
