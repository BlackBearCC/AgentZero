from typing import List, Dict, Any
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
import logging
import aiohttp
from dataclasses import dataclass
import pandas as pd

@dataclass
class TestConfig:
    """测试配置类"""
    name: str  # 配置名称
    use_memory_queue: bool = True  # 是否使用记忆队列
    use_combined_query: bool = False  # 是否使用组合查询
    memory_queue_limit: int = 15  # 记忆队列长度
    llm_model: str = "doubao"  # LLM模型选择
    llm_temperature: float = 0.7  # 温度参数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_memory_queue": self.use_memory_queue,
            "use_combined_query": self.use_combined_query,
            "memory_queue_limit": self.memory_queue_limit,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature
        }

class ComparisonTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.api_version = "v1"
        self.agent_id = "qiyu_001"
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self._logger = logging.getLogger(__name__)
        
        # 预定义测试配置
        self.test_configs = [
            TestConfig(
                name="基础配置",
                use_memory_queue=False,
                use_combined_query=False,
            ),
            TestConfig(
                name="记忆队列",
                use_memory_queue=True,
                use_combined_query=False,
            ),
            TestConfig(
                name="组合查询",
                use_memory_queue=True,
                use_combined_query=True,
            ),
            TestConfig(
                name="DeepSeek模型",
                use_memory_queue=True,
                use_combined_query=True,
                llm_model="deepseek",
            ),
        ]

    async def load_test_cases(self) -> List[Dict[str, Any]]:
        """加载测试用例"""
        test_cases = []
        with open('tests/case/normal_chat_tests.csv', 'r', encoding='utf-8') as f:
            # 跳过注释行和空行
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # 使用第一行作为 header
            if lines:
                reader = csv.DictReader(lines)
                for row in reader:
                    # 只有当所有必需字段都有值时才添加
                    if all(row.get(field) for field in ['test_type', 'topic', 'input']):
                        test_cases.append(row)
                        
                self._logger.info(f"加载了 {len(test_cases)} 个测试用例")
            else:
                self._logger.warning("CSV 文件为空或只包含注释")
                
        return test_cases

    async def stream_chat(self, session: aiohttp.ClientSession, message: str, config: TestConfig) -> str:
        """调用流式对话 API"""
        url = f"{self.base_url}/api/{self.api_version}/chat/{self.agent_id}/stream"
        try:
            # 构建请求数据
            request_data = {
                "message": message,
                "remark": f"配置测试: {config.name}"
            }
            
            # 只有当配置与默认值不同时才添加配置
            if (config.use_memory_queue != True or 
                config.use_combined_query != False or 
                config.memory_queue_limit != 15 or 
                config.llm_model != "doubao" or 
                config.llm_temperature != 0.7):
                request_data["config"] = config.to_dict()

            
            async with session.post(url, json=request_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API错误: {response.status} - {error_text}")
                
                # 读取流式响应
                response_text = ""
                async for chunk in response.content.iter_chunks():
                    chunk_data, _ = chunk
                    chunk_text = chunk_data.decode('utf-8')
                    print(chunk_text, end='', flush=True)
                    response_text += chunk_text
                
                print("\n---")  # 换行分隔符
                print()  # 空行
                
                return {"response": response_text}
                            
        except Exception as e:
            self._logger.error(f"聊天请求失败: {str(e)}")
            return {"error": str(e)}

    def save_comparison_results(self, results: List[Dict[str, Any]], timestamp: str):
        """保存对比结果到CSV"""
        output_dir = Path('tests/results/comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保结果不为空
        if not results:
            self._logger.warning("没有测试结果可保存")
            return
            
        # 保存原始结果
        detailed_file = output_dir / f'detailed_comparison_{timestamp}.csv'
        df = pd.DataFrame(results)
        df.to_csv(detailed_file, index=False, encoding='utf-8')
        
        try:
            # 生成对比表格
            if 'response' not in df.columns:
                self._logger.warning("结果中缺少response字段，使用error字段代替")
                df['response'] = df.apply(
                    lambda row: row.get('response', row.get('error', 'No Response')), 
                    axis=1
                )
            
            pivot_table = pd.pivot_table(
                df,
                values='response',
                index=['test_id', 'topic', 'input'],
                columns=['config_name'],
                aggfunc='first'
            )
            
            comparison_file = output_dir / f'comparison_table_{timestamp}.csv'
            pivot_table.to_csv(comparison_file, encoding='utf-8')
            
        except Exception as e:
            self._logger.error(f"生成对比表格失败: {str(e)}")
            
        self._logger.info(f"详细结果已保存到: {detailed_file}")
        
        # 保存配置信息
        config_info = [config.to_dict() for config in self.test_configs]
        config_file = output_dir / f'test_configs_{timestamp}.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)

    async def run_comparison_tests(self):
        """运行对比测试"""
        self._logger.info("开始执行对比测试...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        test_cases = await self.load_test_cases()
        results = []
        
        async with aiohttp.ClientSession() as session:
            current_topic = None
            
            for test_case in test_cases:
                # 如果话题改变，打印分隔符
                if test_case['topic'] != current_topic:
                    current_topic = test_case['topic']
                    self._logger.info(f"\n=== 开始新话题: {current_topic} ===")
                
                for config in self.test_configs:
                    self._logger.info(f"\n--- 使用配置: {config.name} ---")
                    print(f"\n用户: {test_case['input']}")
                    print(f"[{config.name}] 回复: ", end='', flush=True)
                    
                    # 执行测试
                    response = await self.stream_chat(session, test_case['input'], config)
                    
                    # 记录结果
                    result = {
                        'test_id': f"{test_case['test_type']}_{len(results)}",
                        'topic': test_case['topic'],
                        'input': test_case['input'],
                        'config_name': config.name,
                        'response': response.get('response', ''),
                        'error': response.get('error', ''),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    # 等待一段时间再进行下一次测试
                    await asyncio.sleep(2)
        
        # 保存结果
        self.save_comparison_results(results, timestamp)
        self._logger.info("\n=== 对比测试完成 ===")

async def main():
    """主函数"""
    tester = ComparisonTester()
    await tester.run_comparison_tests()

if __name__ == "__main__":
    asyncio.run(main()) 