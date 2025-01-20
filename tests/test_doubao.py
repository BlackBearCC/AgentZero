import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from src.llm.doubao import DoubaoLLM

async def test_stream():
    """测试豆包 LLM 的流式输出"""
    try:
        # 初始化 LLM
        llm = DoubaoLLM(
            model_name="ep-20241113173739-b6v4g",
            temperature=0.7,
            max_tokens=4096
        )
        
        print("开始流式生成...\n")
        
        # 直接使用流式输出
        async for chunk in llm.astream("用中文介绍一下深度学习"):
            print(chunk, end="", flush=True)
                
        print("\n\n生成完成!")
        
    except Exception as e:
        print(f"测试出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()
    
    # 运行测试
    asyncio.run(test_stream()) 