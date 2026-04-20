"""
检索模块

负责从向量索引中检索与查询最相关的文档块，支持本地和在线两种模式。
"""

from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
from langchain_core.documents import Document

from .vector_indexer import VectorIndexer, create_vector_indexer
from .config import SILICONFLOW_API_KEY


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    document: Document
    score: float
    metadata: Dict[str, Any]
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.document.page_content,
            'score': self.score,
            'metadata': self.metadata,
            'rank': self.rank,
            'source': self.metadata.get('source', '未知来源'),
            'content_preview': self.document.page_content[:200] + '...' if len(self.document.page_content) > 200 else self.document.page_content
        }


@dataclass 
class RetrievalMetrics:
    """检索过程指标"""
    query: str
    total_documents: int
    retrieved_count: int
    search_time: float
    mode: str  # 'local' or 'online'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'query': self.query,
            'total_documents': self.total_documents,
            'retrieved_count': self.retrieved_count,
            'search_time_ms': round(self.search_time * 1000, 2),
            'mode': self.mode
        }


class DocumentRetriever:
    """文档检索器"""
    
    def __init__(self, use_online: bool = False):
        """
        初始化文档检索器
        
        Args:
            use_online: 是否使用在线模式
        """
        self.use_online = use_online
        self.mode = "online" if use_online else "local"
        
        # 初始化向量索引器
        self.indexer: Optional[VectorIndexer] = None
        self._load_or_create_indexer()
    
    def _load_or_create_indexer(self) -> None:
        """加载或创建向量索引器"""
        try:
            self.indexer = create_vector_indexer(use_online=self.use_online)
            
            # 尝试加载已存在的索引
            if not self.indexer.load_index():
                print(f"⚠️  {self.mode}模式索引不存在，需要先构建索引")
                print("   请运行以下命令构建索引：")
                print("   python run_vector_test.py")
                self.indexer = None
            else:
                print(f"✅ {self.mode}模式索引加载成功，包含 {self.indexer.index.ntotal} 个文档块")
                
        except Exception as e:
            print(f"❌ 初始化{self.mode}模式检索器失败: {e}")
            self.indexer = None
    
    def is_ready(self) -> bool:
        """检查检索器是否就绪"""
        return self.indexer is not None and self.indexer.index is not None
    
    def get_index_info(self) -> Dict[str, Any]:
        """获取索引信息"""
        if not self.is_ready():
            return {
                'ready': False,
                'error': '索引未加载'
            }
        
        return {
            'ready': True,
            'mode': self.mode,
            'total_documents': self.indexer.index.ntotal,
            'embedding_dim': self.indexer.index.d,
            'index_type': type(self.indexer.index).__name__
        }
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: Optional[float] = None,
        include_metadata: bool = True
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        检索与查询最相关的文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 分数阈值，低于此分数的结果将被过滤
            include_metadata: 是否包含详细元数据
            
        Returns:
            (检索结果列表, 检索指标)
        """
        if not self.is_ready():
            raise RuntimeError(f"{self.mode}模式索引未就绪，无法进行检索")
        
        if not query.strip():
            raise ValueError("查询不能为空")
        
        # 记录检索开始时间
        start_time = time.time()
        
        try:
            # 调用向量索引器进行搜索
            raw_results = self.indexer.search(query, k=k)
            
            # 应用分数阈值过滤
            if score_threshold is not None:
                raw_results = [r for r in raw_results if r['score'] >= score_threshold]
            
            # 转换为检索结果对象
            results = []
            for result in raw_results:
                retrieval_result = RetrievalResult(
                    document=result['document'],
                    score=result['score'],
                    metadata=result['metadata'] if include_metadata else {'source': result['metadata'].get('source', '未知')},
                    rank=result['rank']
                )
                results.append(retrieval_result)
            
            # 记录检索结束时间并计算指标
            search_time = time.time() - start_time
            
            metrics = RetrievalMetrics(
                query=query,
                total_documents=self.indexer.index.ntotal,
                retrieved_count=len(results),
                search_time=search_time,
                mode=self.mode
            )
            
            return results, metrics
            
        except Exception as e:
            raise RuntimeError(f"检索过程中发生错误: {e}")
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[List[RetrievalResult], RetrievalMetrics]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            k: 每个查询返回的结果数量
            score_threshold: 分数阈值
            
        Returns:
            每个查询的检索结果和指标
        """
        results = []
        
        for query in queries:
            try:
                result = self.retrieve(query, k=k, score_threshold=score_threshold)
                results.append(result)
            except Exception as e:
                # 为失败的查询创建空结果
                empty_results = []
                error_metrics = RetrievalMetrics(
                    query=query,
                    total_documents=0,
                    retrieved_count=0,
                    search_time=0.0,
                    mode=self.mode
                )
                results.append((empty_results, error_metrics))
                print(f"查询 '{query}' 检索失败: {e}")
        
        return results
    
    def get_document_by_rank(self, query: str, rank: int) -> Optional[RetrievalResult]:
        """
        获取指定排名的检索结果
        
        Args:
            query: 查询文本
            rank: 排名 (1-based)
            
        Returns:
            指定排名的检索结果，如果不存在则返回None
        """
        results, _ = self.retrieve(query, k=rank)
        
        if len(results) >= rank:
            return results[rank - 1]
        
        return None
    
    def explain_retrieval(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        解释检索过程，用于可视化和调试
        
        Args:
            query: 查询文本
            k: 检索数量
            
        Returns:
            包含详细检索信息的字典
        """
        if not self.is_ready():
            return {
                'error': f"{self.mode}模式索引未就绪"
            }
        
        results, metrics = self.retrieve(query, k=k, include_metadata=True)
        
        explanation = {
            'query': query,
            'mode': self.mode,
            'index_info': self.get_index_info(),
            'metrics': metrics.to_dict(),
            'results': [result.to_dict() for result in results],
            'analysis': {
                'highest_score': max([r.score for r in results]) if results else 0.0,
                'lowest_score': min([r.score for r in results]) if results else 0.0,
                'score_range': max([r.score for r in results]) - min([r.score for r in results]) if results else 0.0,
                'sources': list(set([r.metadata.get('source', '未知') for r in results]))
            }
        }
        
        return explanation


def create_retriever(use_online: bool = False) -> DocumentRetriever:
    """
    创建文档检索器的工厂函数
    
    Args:
        use_online: 是否使用在线模式
        
    Returns:
        DocumentRetriever实例
    """
    return DocumentRetriever(use_online=use_online)


def auto_select_retriever() -> DocumentRetriever:
    """
    自动选择检索器模式
    
    Returns:
        DocumentRetriever实例，优先使用在线模式（如果可用）
    """
    # 如果有API密钥，优先使用在线模式
    if SILICONFLOW_API_KEY:
        try:
            retriever = create_retriever(use_online=True)
            if retriever.is_ready():
                print("🌐 自动选择在线模式检索器")
                return retriever
        except Exception as e:
            print(f"⚠️  在线模式初始化失败，切换到本地模式: {e}")
    
    # 回退到本地模式
    retriever = create_retriever(use_online=False)
    if retriever.is_ready():
        print("🔧 使用本地模式检索器")
    else:
        print("❌ 本地模式检索器也不可用")
    
    return retriever
