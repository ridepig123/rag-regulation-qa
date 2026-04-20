#!/usr/bin/env python3
"""
RAG文档问答系统主启动脚本

快速启动完整的RAG系统Web界面
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from rag_system.gradio_interface import launch_interface


import subprocess
import time

def kill_port(port):
    """自动关闭占用端口的进程（Windows兼容版本）"""
    import platform
    try:
        if platform.system() == "Windows":
            # Windows: 使用 netstat + taskkill
            result = subprocess.check_output(
                f'netstat -ano | findstr :{port}',
                shell=True
            ).decode()
            
            pids = set()
            for line in result.strip().split('\n'):
                parts = line.strip().split()
                if parts and parts[-1].isdigit():
                    pids.add(parts[-1])
            
            for pid in pids:
                print(f"⚠️  端口 {port} 被占用 (PID: {pid})，正在自动清理...")
                os.system(f"taskkill /F /PID {pid}")
            
            if pids:
                time.sleep(1)
                print(f"✅ 已成功释放端口 {port}")
        else:
            # Linux/macOS
            result = subprocess.check_output(f"lsof -i :{port} -t", shell=True)
            pids = result.decode().strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"⚠️  端口 {port} 被占用 (PID: {pid})，正在自动清理...")
                    os.system(f"kill -9 {pid}")
            if pids:
                time.sleep(1)
                print(f"✅ 已成功释放端口 {port}")

    except subprocess.CalledProcessError:
        # 没有进程占用端口，正常情况
        pass
    except Exception as e:
        print(f"⚠️  尝试释放端口 {port} 时出错: {e}")

def main():
    """主启动函数"""
    print("=" * 60)
    print("🤖 RAG文档问答系统")
    print("=" * 60)
    print()
    print("🔧 系统特性:")
    print("  • 双模式运行: 本地Ollama + 在线SiliconFlow")
    print("  • 智能文档解析: PDF、DOCX格式支持")
    print("  • 实时处理可视化: 分块、向量化全程展示")
    print("  • 语义检索: FAISS高效相似度搜索")
    print("  • 透明RAG流程: 检索结果、提示词全面展示")
    print()
    print("📋 使用说明:")
    print("  1. 选择运行模式（本地/在线）")
    print("  2. 检查系统配置状态")
    print("  3. 上传PDF或DOCX文档")
    print("  4. 观察实时处理过程")
    print("  5. 配置检索参数")
    print("  6. 开始智能问答")
    print()
    print("⚠️  注意事项:")
    print("  • 本地模式需要运行Ollama服务")
    print("  • 在线模式需要配置SILICONFLOW_API_KEY")
    print("  • 建议使用虚拟环境运行")
    print()
    
    # 自动处理端口冲突
    kill_port(7860)
    
    print("🚀 正在启动Web界面...")
    print()
    
    try:
        # 启动界面
        launch_interface(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n👋 感谢使用RAG文档问答系统！")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n🔧 故障排除:")
        print("  1. 检查是否安装了所有依赖: pip install -r requirements.txt")
        print("  2. 确认Python版本 >= 3.8")
        print("  3. 检查端口7860是否被占用")
        print("  4. 查看详细错误日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
