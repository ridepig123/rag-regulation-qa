"""
Gradio可视化Web界面

提供完整的RAG系统交互界面，包括文档上传、实时处理可视化、问答交互等功能。
"""

import gradio as gr
import os
import time
import shutil
import html
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Generator
from pathlib import Path

from .config import (
    SILICONFLOW_API_KEY, BASE_DIR,
    DEFAULT_RETRIEVAL_K, DEFAULT_RETRIEVAL_THRESHOLD,
    CHUNK_PREVIEW_LENGTH, CHUNK_SIZE, CHUNK_OVERLAP,
    LOCAL_LLM_MODEL, ONLINE_LLM_MODEL
)
from .document_parser import load_document
from .text_chunker import split_documents
from .vector_indexer import create_vector_indexer
from .retriever import create_retriever
from .generator import create_generator

# 统一日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class RAGInterface:
    """RAG系统界面管理类"""

    def __init__(self):
        self.current_mode = "local"
        self.retrieval_config = {
            "k": DEFAULT_RETRIEVAL_K,
            "threshold": DEFAULT_RETRIEVAL_THRESHOLD
        }
        self.documents = []
        self.indexer = None
        self.retriever = None
        self.generator = None

        source_doc_dir = BASE_DIR / "data" / "source_documents"
        os.makedirs(source_doc_dir, exist_ok=True)

    def check_system_status(self, mode: str) -> Tuple[str, bool]:
        """检查系统组件是否就绪"""
        use_online = mode in ["online", "siliconflow"]
        try:
            if use_online:
                if not SILICONFLOW_API_KEY:
                    return "⚠️ SiliconFlow: 未配置API密钥", False
                test_indexer = create_vector_indexer(use_online=True)
                test_generator = create_generator(use_online=True)
                if test_indexer and test_generator:
                    return "✅ SiliconFlow: 嵌入模型✅ | LLM✅ | 就绪", True
                return "❌ SiliconFlow: 组件初始化失败", False
            else:
                test_indexer = create_vector_indexer(use_online=False)
                test_generator = create_generator(use_online=False)
                if test_indexer and test_generator.is_ready():
                    return "✅ Ollama: 嵌入模型✅ | LLM✅ | 就绪", True
                return "⚠️ Ollama: 服务未运行或模型未安装", False
        except Exception as e:
            return f"❌ 系统检查失败: {str(e)[:60]}", False

    def process_documents(self, files: List, mode: str) -> Generator[Tuple[str, str, str], None, None]:
        """
        处理上传的文档，实时输出进度日志。

        Yields:
            (log_content, block_details_html, status_message)
        """
        if not files:
            yield "❌ 请先选择要上传的文件", "", "等待文件上传"
            return

        self.current_mode = mode
        self.documents = []
        log_lines = []

        def log(msg: str):
            log_lines.append(msg)
            logger.info(msg)

        log(f"[{time.strftime('%H:%M:%S')}] 开始处理，共 {len(files)} 个文件")
        yield "\n".join(log_lines), "", "解析文档中..."

        # ── 阶段1：文档解析 ──────────────────────────────
        all_raw_documents = []
        failed_files = []

        for i, file in enumerate(files, 1):
            filename = os.path.basename(file.name)
            dest_path = BASE_DIR / "data" / "source_documents" / filename
            try:
                shutil.copy2(file.name, dest_path)
                documents = load_document(str(dest_path))
                all_raw_documents.extend(documents)
                log(f"[{time.strftime('%H:%M:%S')}] ✅ [{i}/{len(files)}] {filename} → {len(documents)} 个原始块")
            except Exception as e:
                failed_files.append(filename)
                log(f"[{time.strftime('%H:%M:%S')}] ❌ [{i}/{len(files)}] {filename} 解析失败: {e}")
                logger.error(f"文档解析失败: {filename}", exc_info=True)

            yield "\n".join(log_lines), "", f"解析中 {i}/{len(files)}"

        if failed_files:
            log(f"[{time.strftime('%H:%M:%S')}] ⚠️ 失败文件: {', '.join(failed_files)}")

        if not all_raw_documents:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ 没有成功解析任何文档，终止处理")
            yield "\n".join(log_lines), "", "❌ 处理失败"
            return

        # ── 阶段2：分块 ──────────────────────────────────
        log(f"[{time.strftime('%H:%M:%S')}] 开始分块，原始块数: {len(all_raw_documents)}")
        yield "\n".join(log_lines), "", "分块中..."

        split_docs = split_documents(all_raw_documents)
        self.documents = split_docs
        block_details = self._generate_block_details(split_docs)
        log(f"[{time.strftime('%H:%M:%S')}] ✅ 分块完成，共 {len(split_docs)} 个块")
        yield "\n".join(log_lines), block_details, "向量化中..."

        # ── 阶段3：向量化与索引 ───────────────────────────
        use_online = mode in ["online", "siliconflow"]
        mode_name = "在线" if use_online else "本地"

        log(f"[{time.strftime('%H:%M:%S')}] 加载 {mode_name} 嵌入模型...")
        yield "\n".join(log_lines), block_details, "加载模型..."

        try:
            self.indexer = create_vector_indexer(use_online=use_online)
            self.indexer.build_index(split_docs)
            self.indexer.save_index()
            log(f"[{time.strftime('%H:%M:%S')}] ✅ 索引构建完成 | 维度: {self.indexer.index.d} | 向量数: {len(split_docs)}")

            self.retriever = create_retriever(use_online=use_online)
            self.generator = create_generator(use_online=use_online)
            log(f"[{time.strftime('%H:%M:%S')}] ✅ 检索器与生成器初始化完成")
        except Exception as e:
            log(f"[{time.strftime('%H:%M:%S')}] ❌ 索引构建失败: {e}")
            logger.error("索引构建失败", exc_info=True)
            yield "\n".join(log_lines), block_details, "❌ 索引失败"
            return

        final_status = f"✅ 完成: {len(split_docs)} 个块已就绪"
        if failed_files:
            final_status += f"（{len(failed_files)} 个文件失败）"
        log(f"[{time.strftime('%H:%M:%S')}] {final_status}")
        yield "\n".join(log_lines), block_details, final_status

    def _generate_block_details(self, documents: List) -> str:
        """生成文档分块的HTML展示"""
        if not documents:
            return "<div style='padding: 20px; text-align: center; color: #6c757d;'>暂无文档块</div>"

        source_stats: Dict[str, List] = {}
        for doc in documents:
            source = doc.metadata.get("source", "未知来源")
            source_stats.setdefault(source, []).append(doc)

        html_parts = [f"""
        <div style='padding: 10px;'>
            <div style='margin-bottom: 15px; font-weight: bold; color: #495057;'>
                📋 文档分块详情（共 {len(documents)} 个块）
            </div>
        """]

        for source, docs in source_stats.items():
            html_parts.append(f"<div style='margin: 5px 0; color: #6c757d;'>📄 {source}: {len(docs)} 个块</div>")

        html_parts.append("<hr style='margin: 15px 0; border: 1px solid #dee2e6;'>")

        for i, doc in enumerate(documents[:10], 1):
            source = doc.metadata.get("source", "未知来源")
            preview = doc.page_content[:CHUNK_PREVIEW_LENGTH].replace("\n", " ")
            if len(doc.page_content) > CHUNK_PREVIEW_LENGTH:
                preview += "..."
            html_parts.append(f"""
            <div class='chunk-item' style='border: 2px solid #6c757d; border-radius: 8px; padding: 12px; margin: 8px 0; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='font-weight: bold; color: #495057; margin-bottom: 8px;'>
                    块 {i}：[{source}]（{len(doc.page_content)} 字符）
                </div>
                <div style='color: #6c757d; font-size: 14px; line-height: 1.4;'>{preview}</div>
            </div>
            """)

        if len(documents) > 10:
            html_parts.append(f"<div style='text-align: center; color: #6c757d; margin-top: 15px;'>... 还有 {len(documents) - 10} 个块</div>")

        html_parts.append("</div>")
        return "".join(html_parts)

    def query(self, query_text: str) -> Tuple[str, str]:
        """
        执行检索与生成，返回答案HTML和召回片段HTML。

        Returns:
            (answer_html, retrieval_html)
        """
        if not query_text.strip():
            return self._info_html("请输入问题"), ""

        if not self.retriever or not self.generator:
            return self._error_html("系统未就绪，请先上传并处理文档"), ""

        try:
            retrieval_results, retrieval_metrics = self.retriever.retrieve(
                query_text,
                k=self.retrieval_config["k"],
                score_threshold=self.retrieval_config["threshold"]
            )
            logger.info(f"检索完成: {len(retrieval_results)} 个片段，用时 {retrieval_metrics.search_time*1000:.1f}ms")

            generation_result = self.generator.generate_answer(query_text, retrieval_results)
            logger.info(f"生成完成，用时 {generation_result.generation_time:.2f}s")

            return self._format_answer(generation_result, retrieval_results), self._format_retrieval(retrieval_results)

        except Exception as e:
            logger.error("查询处理失败", exc_info=True)
            return self._error_html(f"处理失败: {e}"), ""

    def _format_answer(self, generation_result, retrieval_results) -> str:
        """格式化答案为HTML"""
        answer_text = str(generation_result.answer) if generation_result.answer else "未生成答案"

        # 清理Markdown格式
        answer_cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', answer_text)
        answer_cleaned = re.sub(r'\*(.*?)\*', r'\1', answer_cleaned)
        answer_cleaned = re.sub(r'^- ', '• ', answer_cleaned, flags=re.MULTILINE)
        answer_cleaned = re.sub(r'#{1,6}\s*', '', answer_cleaned)
        answer_escaped = html.escape(answer_cleaned)

        # 思维链
        thinking_html = ""
        if generation_result.thinking_chain:
            chain = generation_result.thinking_chain.strip()
            display = chain[:500] + ("..." if len(chain) > 500 else "")
            thinking_html = f"""
            <div style='margin-bottom: 12px; padding: 10px; background: #f0f8ff; border-left: 3px solid #007bff; border-radius: 4px; font-size: 13px; color: #555; line-height: 1.5;'>
                <strong>🧠 思维过程：</strong><br/>{html.escape(display)}
            </div>"""

        # 来源标签
        sources = sorted({doc["source"] for doc in generation_result.source_documents})
        sources_html = "".join(
            f"<span style='background:#e3f2fd;color:#1976d2;padding:3px 10px;border-radius:12px;font-size:13px;margin:2px;display:inline-block;'>{s}</span>"
            for s in sources
        )

        mode_label = "🌐 在线" if self.current_mode in ["online", "siliconflow"] else "🔧 本地"
        token_info = ""
        if generation_result.prompt_tokens:
            token_info = f" | Token: {generation_result.prompt_tokens}→{generation_result.completion_tokens}"

        return f"""
        <div style='padding: 16px; background: white; border: 1px solid #dee2e6; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);'>
            <h4 style='color: #007bff; margin: 0 0 12px 0;'>💡 智能回答</h4>
            {thinking_html}
            <div style='line-height: 1.7; color: #333; font-size: 14px; white-space: pre-wrap;'>{answer_escaped}</div>
            <hr style='margin: 12px 0; border: none; border-top: 1px solid #eee;'/>
            <div style='margin-bottom: 8px;'>{sources_html}</div>
            <small style='color: #999; font-size: 12px;'>
                ⏱️ {generation_result.generation_time:.1f}s | 📊 {len(retrieval_results)} 个片段 | {mode_label}{token_info}
            </small>
        </div>"""

    def _format_retrieval(self, retrieval_results) -> str:
        """格式化召回片段为HTML"""
        if not retrieval_results:
            return "<div style='padding: 20px; text-align: center; color: #6c757d;'>未找到相关片段</div>"

        parts = [f"<div style='padding: 10px;'><div style='margin-bottom: 15px; font-weight: bold; color: #495057;'>🎯 召回片段（共 {len(retrieval_results)} 个）</div>"]

        for i, result in enumerate(retrieval_results, 1):
            score = result.score
            source = result.metadata.get("source", "未知来源")
            content_escaped = html.escape(result.document.page_content)
            progress = int(score * 100)

            parts.append(f"""
            <div style='border: 2px solid #6c757d; border-radius: 8px; padding: 12px; margin: 8px 0; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='font-weight: bold; color: #495057; margin-bottom: 6px;'>
                    片段 {i}：[{source}]
                    <span style='color: #28a745; font-size: 13px; margin-left: 8px;'>相似度: {score:.3f}</span>
                </div>
                <div style='width:100%;background:#e9ecef;border-radius:10px;height:6px;margin:6px 0;'>
                    <div style='width:{progress}%;background:#28a745;height:100%;border-radius:10px;'></div>
                </div>
                <div style='color:#6c757d;font-size:13px;line-height:1.4;margin-top:8px;max-height:120px;overflow-y:auto;border:1px solid #dee2e6;border-radius:4px;padding:8px;background:#f8f9fa;font-family:Consolas,monospace;white-space:pre-wrap;word-break:break-all;'>
                    {content_escaped}
                </div>
            </div>""")

        parts.append("</div>")
        return "".join(parts)

    @staticmethod
    def _error_html(msg: str) -> str:
        return f"<div style='background:#f8d7da;border:1px solid #f5c6cb;border-radius:8px;padding:15px;color:#721c24;'>❌ {html.escape(msg)}</div>"

    @staticmethod
    def _info_html(msg: str) -> str:
        return f"<div style='background:#f8f9fa;border:1px solid #e9ecef;border-radius:6px;padding:15px;color:#6c757d;text-align:center;'>{html.escape(msg)}</div>"


def create_interface() -> gr.Blocks:
    """创建Gradio界面"""

    rag = RAGInterface()

    css = """
    * { font-family: "SimSun", "宋体", serif !important; }
    .main-title { text-align: center !important; font-size: 24px !important; font-weight: bold !important; margin: 20px 0 10px !important; }
    .chunk-container { height: 300px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; background: #f8f9fa; }
    .chunk-item { border: 2px solid #6c757d; border-radius: 8px; padding: 12px; margin: 8px 0; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .retrieval-container { height: 800px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; background: #f8f9fa; }
    .left-panel { min-height: 85vh !important; height: auto !important; overflow-y: auto !important; padding: 8px !important; background: #fafbfc !important; border-radius: 8px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; }
    #answer_area { margin: 10px 0; min-height: 100px; }
    .config-info { position: fixed; top: 10px; right: 10px; background: rgba(255,255,255,0.95); border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px 12px; font-size: 11px; color: #6b7280; box-shadow: 0 2px 4px rgba(0,0,0,0.1); z-index: 1000; max-width: 200px; }
    .log-box textarea { font-family: Consolas, monospace !important; font-size: 12px !important; line-height: 1.5 !important; }
    """

    with gr.Blocks(title="企业规章制度RAG问答系统", css=css, theme=gr.themes.Soft()) as interface:

        gr.HTML('<h1 class="main-title">🤖 企业规章制度原生RAG问答系统</h1>')

        gr.HTML(f"""<div class='config-info'>
            <div style='font-weight:bold;margin-bottom:4px;color:#374151;'>📊 系统配置</div>
            <div style='display:flex;justify-content:space-between;margin:2px 0;'><span>分块大小:</span><span>{CHUNK_SIZE}</span></div>
            <div style='display:flex;justify-content:space-between;margin:2px 0;'><span>重叠长度:</span><span>{CHUNK_OVERLAP}</span></div>
            <div style='display:flex;justify-content:space-between;margin:2px 0;'><span>检索数量:</span><span>{DEFAULT_RETRIEVAL_K}</span></div>
            <div style='display:flex;justify-content:space-between;margin:2px 0;'><span>相似度阈值:</span><span>{DEFAULT_RETRIEVAL_THRESHOLD}</span></div>
        </div>""")

        with gr.Row():
            # ── 左侧操作区 ────────────────────────────────
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                mode_radio = gr.Radio(
                    choices=["ollama", "siliconflow"],
                    value="ollama",
                    label="运行模式"
                )
                system_status = gr.Markdown("⏳ 检测中...")

                file_upload = gr.File(
                    label="上传文档（支持 PDF、DOCX）",
                    file_count="multiple",
                    file_types=[".pdf", ".docx"],
                    height=80
                )
                process_btn = gr.Button("📂 处理文档", variant="primary", size="sm")

                # 处理日志：默认隐藏，处理完成后自动显示
                process_log = gr.Textbox(
                    label="处理日志",
                    lines=6,
                    interactive=False,
                    visible=False,
                    elem_classes=["log-box"]
                )

                gr.Markdown("---")

                query_input = gr.Textbox(
                    label="提问",
                    placeholder="输入问题...",
                    lines=2
                )
                submit_btn = gr.Button("🔍 提交问题", variant="primary", size="sm")

                answer_output = gr.HTML(
                    value=RAGInterface._info_html("请先上传文档，然后输入问题开始问答"),
                    elem_id="answer_area"
                )

            # ── 右侧可视化区 ──────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("## 📊 文档可视化")

                with gr.Group():
                    gr.Markdown("### 📋 文档分块详情")
                    block_details = gr.HTML(
                        value="<div class='chunk-container' style='text-align:center;padding:20px;color:#6c757d;'>请先上传并处理文档</div>",
                        elem_classes=["chunk-container"]
                    )

                with gr.Group():
                    gr.Markdown("### 🎯 召回片段详情")
                    retrieval_details = gr.HTML(
                        value="<div style='padding:20px;text-align:center;color:#6c757d;'>等待问答完成后显示召回片段</div>",
                        elem_classes=["retrieval-container"]
                    )

        # ── 事件绑定 ──────────────────────────────────────

        def on_mode_change(mode):
            status, _ = rag.check_system_status(mode)
            return status

        def on_process(files, mode):
            """处理文档，收集完整日志后统一展示"""
            log_text = ""
            block_html = ""

            for log_content, block_details_html, _ in rag.process_documents(files, mode):
                log_text = log_content
                if block_details_html:
                    block_html = block_details_html

            if not block_html:
                block_html = "<div class='chunk-container' style='text-align:center;padding:20px;color:#6c757d;'>处理失败，请查看日志</div>"

            return block_html, gr.update(value=log_text, visible=True)

        def on_query(query_text):
            answer_html, retrieval_html = rag.query(query_text)
            return answer_html, retrieval_html

        mode_radio.change(fn=on_mode_change, inputs=[mode_radio], outputs=[system_status])
        interface.load(fn=on_mode_change, inputs=[mode_radio], outputs=[system_status])

        process_btn.click(
            fn=on_process,
            inputs=[file_upload, mode_radio],
            outputs=[block_details, process_log]
        )

        submit_btn.click(
            fn=on_query,
            inputs=[query_input],
            outputs=[answer_output, retrieval_details]
        )

    return interface


def launch_interface(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """启动Gradio界面"""
    interface = create_interface()
    logger.info(f"启动RAG文档问答系统，访问地址: http://localhost:{server_port}")
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    launch_interface()