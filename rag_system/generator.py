"""
生成模块

负责基于检索到的文档内容，调用大语言模型生成最终答案，支持本地和在线两种模式。
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import json
from dataclasses import dataclass
import requests
import ollama

from .retriever import RetrievalResult
from .config import (
    SILICONFLOW_API_KEY,
    LOCAL_LLM_MODEL,
    ONLINE_LLM_MODEL,
    OLLAMA_BASE_URL,
    SILICONFLOW_BASE_URL
)


@dataclass
class GenerationResult:
    """生成结果数据类"""
    answer: str
    query: str
    source_documents: List[Dict[str, Any]]
    generation_time: float
    mode: str  # 'local' or 'online'
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    thinking_chain: Optional[str] = None  # 新增：思维链
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'answer': self.answer,
            'query': self.query,
            'source_documents': self.source_documents,
            'generation_time_ms': round(self.generation_time * 1000, 2),
            'mode': self.mode,
            'model_name': self.model_name,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': (self.prompt_tokens + self.completion_tokens) if self.prompt_tokens and self.completion_tokens else None
        }


class PromptTemplate:
    """提示词模板管理"""
    
    DEFAULT_SYSTEM_PROMPT = """你是一个专业的企业制度解答助手。你的任务是基于提供的公司文档内容，准确回答用户的问题。

请遵循以下原则：
1. 仅基于提供的文档内容回答问题
2. 如果文档中没有相关信息，请明确说明"根据提供的文档，没有找到相关信息"
3. 回答要准确、简洁、有条理
4. 如果涉及具体数字、标准或流程，请准确引用
5. 可以适当引用文档来源以增加可信度
6. 最终回答请控制在500字以内，确保信息精炼且完整"""

    DEFAULT_USER_TEMPLATE = """基于以下文档内容，请回答用户的问题。

**用户问题：**
{query}

**相关文档内容：**
{context}

**请提供准确、有条理的回答：**"""

    RAG_ASSISTANT_TEMPLATE = """我来基于提供的文档内容回答您的问题。

{answer}

---
*以上回答基于公司内部文档，如需了解更多详细信息，请参考相关制度文件。*"""

    @classmethod
    def format_context(cls, retrieval_results: List[RetrievalResult]) -> str:
        """格式化检索结果为上下文"""
        if not retrieval_results:
            return "没有找到相关文档内容。"
        
        context_parts = []
        for i, result in enumerate(retrieval_results, 1):
            source = result.metadata.get('source', '未知来源')
            content = result.document.page_content.strip()
            score = result.score
            
            context_part = f"""文档片段 {i} (来源: {source}, 相关度: {score:.3f}):
{content}"""
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    @classmethod
    def build_prompt(
        cls, 
        query: str, 
        retrieval_results: List[RetrievalResult],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        构建完整的提示词
        
        Args:
            query: 用户查询
            retrieval_results: 检索结果
            system_prompt: 自定义系统提示词
            user_template: 自定义用户提示词模板
            
        Returns:
            (system_prompt, user_prompt)
        """
        system = system_prompt or cls.DEFAULT_SYSTEM_PROMPT
        user_template = user_template or cls.DEFAULT_USER_TEMPLATE
        
        context = cls.format_context(retrieval_results)
        user_prompt = user_template.format(query=query, context=context)
        
        return system, user_prompt


class LLMClient:
    """大语言模型客户端基类"""
    
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Tuple[str, Optional[int], Optional[int]]:
        """
        生成回答
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            (generated_text, prompt_tokens, completion_tokens)
        """
        raise NotImplementedError


class LocalLLMClient(LLMClient):
    """本地LLM客户端（基于Ollama）"""
    
    def __init__(self, model_name: str = LOCAL_LLM_MODEL, base_url: str = OLLAMA_BASE_URL):
        """
        初始化本地LLM客户端
        
        Args:
            model_name: 模型名称
            base_url: Ollama服务地址
        """
        self.model_name = model_name
        self.base_url = base_url
        
        # 验证Ollama连接
        try:
            client = ollama.Client(host=base_url)
            models_response = client.list()
            
            # 检查模型是否可用
            # 兼容不同版本的ollama返回格式
            if hasattr(models_response, 'models'):
                # 新版本返回对象
                available_models = [model.model for model in models_response.models]
            else:
                # 旧版本返回字典
                available_models = [model['name'] for model in models_response['models']]
                
            if model_name not in available_models:
                print(f"⚠️  模型 {model_name} 未在Ollama中找到")
                print(f"   可用模型: {', '.join(available_models)}")
                print(f"   请运行: ollama run {model_name}")
            else:
                print(f"✅ 本地LLM模型 {model_name} 连接成功")
                
        except Exception as e:
            print(f"❌ 连接Ollama失败: {e}")
            print(f"   请确保Ollama服务运行在 {base_url}")
    
    def _extract_thinking_and_answer(self, text: str) -> Tuple[str, str]:
        """
        提取思维链和最终答案
        
        Args:
            text: 原始响应文本
            
        Returns:
            (thinking_chain, clean_answer): 思维链和清理后的答案
        """
        import re
        
        # 提取思维链内容
        thinking_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        thinking_chain = ""
        if thinking_match:
            thinking_chain = thinking_match.group(1).strip()
        
        # 移除 <think>...</think> 标签及其内容，保留答案
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 移除其他可能的XML标签
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        
        # 清理多余的空白字符
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
        clean_text = clean_text.strip()
        
        return thinking_chain, clean_text
    
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Tuple[str, Optional[int], Optional[int]]:
        """使用Ollama生成回答"""
        try:
            client = ollama.Client(host=self.base_url)
            
            response = client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.1,  # 较低温度以获得更一致的回答
                    'top_p': 0.9,
                    'top_k': 40
                }
            )
            
            generated_text = response['message']['content']
            
            # 提取思维链和答案
            thinking_chain, cleaned_text = self._extract_thinking_and_answer(generated_text)
            
            # 如果有思维链，临时存储（后续会在GenerationResult中返回）
            if hasattr(self, '_last_thinking_chain'):
                self._last_thinking_chain = thinking_chain
            else:
                self._last_thinking_chain = thinking_chain
            
            # Ollama暂不提供token统计，返回None
            return cleaned_text, None, None
            
        except Exception as e:
            raise RuntimeError(f"本地LLM生成失败: {e}")


class OnlineLLMClient(LLMClient):
    """在线LLM客户端（基于SiliconFlow API）"""
    
    def __init__(self, api_key: str = SILICONFLOW_API_KEY, model_name: str = ONLINE_LLM_MODEL):
        """
        初始化在线LLM客户端
        
        Args:
            api_key: API密钥
            model_name: 模型名称
        """
        if not api_key:
            raise ValueError("API密钥不能为空，请设置SILICONFLOW_API_KEY环境变量")
        
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = SILICONFLOW_BASE_URL
        print(f"✅ 在线LLM模型 {model_name} 初始化成功")
    
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Tuple[str, Optional[int], Optional[int]]:
        """使用在线API生成回答"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 2000,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=300  # 增加超时时间到5分钟，适配推理模型
            )
            response.raise_for_status()
            
            result = response.json()
            
            generated_text = result['choices'][0]['message']['content']
            
            # 提取token使用情况
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens')
            completion_tokens = usage.get('completion_tokens')
            
            return generated_text, prompt_tokens, completion_tokens
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"在线LLM API调用失败: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"解析在线LLM API响应失败: {e}")
        except Exception as e:
            raise RuntimeError(f"在线LLM API调用失败: {e}")


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, use_online: bool = False):
        """
        初始化答案生成器
        
        Args:
            use_online: 是否使用在线模式
        """
        self.use_online = use_online
        self.mode = "online" if use_online else "local"
        
        # 初始化LLM客户端
        try:
            if use_online:
                self.llm_client = OnlineLLMClient()
            else:
                self.llm_client = LocalLLMClient()
                
            self.model_name = self.llm_client.model_name
            
        except Exception as e:
            print(f"❌ 初始化{self.mode}LLM客户端失败: {e}")
            self.llm_client = None
            self.model_name = "未知"
    
    def is_ready(self) -> bool:
        """检查生成器是否就绪"""
        return self.llm_client is not None
    
    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        include_sources: bool = True
    ) -> GenerationResult:
        """
        生成答案
        
        Args:
            query: 用户查询
            retrieval_results: 检索结果
            system_prompt: 自定义系统提示词
            user_template: 自定义用户提示词模板
            include_sources: 是否在结果中包含源文档信息
            
        Returns:
            生成结果
        """
        if not self.is_ready():
            raise RuntimeError(f"{self.mode}模式LLM客户端未就绪")
        
        if not query.strip():
            raise ValueError("查询不能为空")
        
        # 构建提示词
        system, user_prompt = PromptTemplate.build_prompt(
            query, retrieval_results, system_prompt, user_template
        )
        
        # 记录生成开始时间
        start_time = time.time()
        
        try:
            # 调用LLM生成答案
            answer, prompt_tokens, completion_tokens = self.llm_client.generate(
                system, user_prompt
            )
            
            # 记录生成结束时间
            generation_time = time.time() - start_time
            
            # 准备源文档信息
            source_documents = []
            if include_sources:
                for result in retrieval_results:
                    source_doc = {
                        'content': result.document.page_content,
                        'source': result.metadata.get('source', '未知来源'),
                        'score': result.score,
                        'rank': result.rank
                    }
                    source_documents.append(source_doc)
            
            # 获取思维链（如果存在）
            thinking_chain = None
            if hasattr(self.llm_client, '_last_thinking_chain'):
                thinking_chain = self.llm_client._last_thinking_chain
            
            return GenerationResult(
                answer=answer.strip(),
                query=query,
                source_documents=source_documents,
                generation_time=generation_time,
                mode=self.mode,
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thinking_chain=thinking_chain
            )
            
        except Exception as e:
            raise RuntimeError(f"答案生成过程中发生错误: {e}")
    
    def generate_with_explanation(
        self,
        query: str,
        retrieval_results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        生成带详细解释的答案
        
        Args:
            query: 用户查询
            retrieval_results: 检索结果
            
        Returns:
            包含答案、提示词、源文档等详细信息的字典
        """
        if not self.is_ready():
            return {
                'error': f"{self.mode}模式LLM客户端未就绪"
            }
        
        # 构建提示词
        system_prompt, user_prompt = PromptTemplate.build_prompt(query, retrieval_results)
        
        # 生成答案
        result = self.generate_answer(query, retrieval_results)
        
        # 构建详细解释
        explanation = {
            'query': query,
            'mode': self.mode,
            'model_name': self.model_name,
            'answer': result.answer,
            'generation_metrics': {
                'generation_time_ms': result.generation_time * 1000,
                'prompt_tokens': result.prompt_tokens,
                'completion_tokens': result.completion_tokens
            },
            'prompts': {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt
            },
            'source_documents': result.source_documents,
            'retrieval_summary': {
                'total_retrieved': len(retrieval_results),
                'sources': list(set([r.metadata.get('source', '未知') for r in retrieval_results])),
                'score_range': {
                    'highest': max([r.score for r in retrieval_results]) if retrieval_results else 0.0,
                    'lowest': min([r.score for r in retrieval_results]) if retrieval_results else 0.0
                }
            }
        }
        
        return explanation


def create_generator(use_online: bool = False) -> AnswerGenerator:
    """
    创建答案生成器的工厂函数
    
    Args:
        use_online: 是否使用在线模式
        
    Returns:
        AnswerGenerator实例
    """
    return AnswerGenerator(use_online=use_online)


def auto_select_generator() -> AnswerGenerator:
    """
    自动选择生成器模式
    
    Returns:
        AnswerGenerator实例，优先使用在线模式（如果可用）
    """
    # 如果有API密钥，优先使用在线模式
    if SILICONFLOW_API_KEY:
        try:
            generator = create_generator(use_online=True)
            if generator.is_ready():
                print("🌐 自动选择在线模式生成器")
                return generator
        except Exception as e:
            print(f"⚠️  在线模式初始化失败，切换到本地模式: {e}")
    
    # 回退到本地模式
    generator = create_generator(use_online=False)
    if generator.is_ready():
        print("🔧 使用本地模式生成器")
    else:
        print("❌ 本地模式生成器也不可用")
    
    return generator
