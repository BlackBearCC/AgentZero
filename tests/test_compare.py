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
from openpyxl.styles import Alignment, Font

@dataclass
class TestConfig:
    """测试配置类"""
    name: str  # 配置名称
    enable_memory_recall: bool = True  # 新增记忆召回开关
    use_memory_queue: bool = True  # 是否使用记忆队列
    use_combined_query: bool = False  # 是否使用组合查询
    use_event_summary: bool = True  # 新增事件概要开关
    memory_queue_limit: int = 15  # 记忆队列长度
    llm_model: str = "doubao"  # LLM模型选择
    llm_temperature: float = 0.7  # 温度参数
    max_parallel: int = 3  # 新增最大并行数控制
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_memory_recall": self.enable_memory_recall,
            "use_memory_queue": self.use_memory_queue,
            "use_combined_query": self.use_combined_query,
            "use_event_summary": self.use_event_summary,
            "memory_queue_limit": self.memory_queue_limit,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature
        }

class ComparisonTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.api_version = "v1"
        self.agent_id = "qiyu_001"
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path('tests/results/comparison')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 新的文件路径
        self.comparison_table = self.output_dir / f'comparison_table_{self.timestamp}.xlsx'
        self.raw_data_file = self.output_dir / f'raw_data_{self.timestamp}.csv'
        self.config_file = self.output_dir / f'test_configs_{self.timestamp}.json'
        
        # 存储结果的数据结构
        self.results = []
        self.comparison_data = {}  # 用于存储对比数据 {test_id: {config_name: response}}
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self._logger = logging.getLogger(__name__)
        
        # 更新测试配置
        self.test_configs = [
            TestConfig(name="无记忆聊天", 
                      enable_memory_recall=False,
                      use_memory_queue=False,
                      use_combined_query=False,
                      use_event_summary=False,),
            TestConfig(name="记忆召回-无队列", 
                      enable_memory_recall=True,
                      use_memory_queue=False,
                      use_combined_query=False,
                      use_event_summary=False),
            TestConfig(name="记忆召回-有队列", 
                      enable_memory_recall=True,
                      use_memory_queue=True,
                      use_combined_query=False,
                      use_event_summary=False),

            # TestConfig(name="事件概要", 
            #           enable_memory_recall=False,
            #           use_memory_queue=False,
            #           use_combined_query=False, 
            #           use_event_summary=True),

            # TestConfig(name="事件概要-记忆召回", 
            #           enable_memory_recall=True,
            #           use_memory_queue=False,
            #           use_combined_query=False,
            #           use_event_summary=True),
        ]
        
        # 保存配置信息
        self._save_config_info()
        self.semaphore = asyncio.Semaphore(3)  # 默认最大并行数3
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化Excel文件
        self._init_excel_file()

    def _save_config_info(self):
        """保存配置信息到JSON文件"""
        config_info = [config.to_dict() for config in self.test_configs]
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)

    def _init_excel_file(self):
        """初始化Excel文件并写入表头"""
        columns = ['test_id', 'topic', 'input', 'timestamp'] + [c.name for c in self.test_configs]
        init_df = pd.DataFrame(columns=columns)
        with pd.ExcelWriter(self.comparison_table, engine='openpyxl') as writer:
            init_df.to_excel(writer, sheet_name='对比结果', index=False)

    def save_comparison_results(self):
        """保存对比结果"""
        if not self.results:
            return

        try:
            # 构建表格行
            rows = []
            columns = ['用户输入'] + [c.name for c in self.test_configs]
            
            # 添加表头（只在第一次写入）
            if not Path(self.comparison_table).exists():
                rows.append(columns)
            
            # 按测试用例分组结果
            grouped_results = {}
            for result in self.results:
                key = (result['test_id'], result['topic'], result['input'])
                if key not in grouped_results:
                    grouped_results[key] = []
                grouped_results[key].append(result)

            # 添加数据行
            for key in grouped_results:
                test_case = grouped_results[key]
                input_text = test_case[0]['input']
                
                # 用户输入行
                rows.append([input_text] + [''] * len(self.test_configs))
                
                # 回复内容行
                response_row = ['→ 回复']
                for config in self.test_configs:
                    response = next((r['response'] for r in test_case if r['config_name'] == config.name), '')
                    response_row.append(response)
                rows.append(response_row)
                
                # 实体记忆行
                memory_row = ['→ 记忆']
                for config in self.test_configs:
                    memory = next((r['metadata'] for r in test_case if r['config_name'] == config.name), '')
                    memory_row.append(memory)
                rows.append(memory_row)
                
                # 添加空行分隔
                rows.append([''] * (len(self.test_configs) + 1))

            # 转换为DataFrame
            df = pd.DataFrame(rows[1:], columns=columns if not Path(self.comparison_table).exists() else None)

            # 写入Excel
            with pd.ExcelWriter(
                self.comparison_table, 
                engine='openpyxl', 
                mode='a' if Path(self.comparison_table).exists() else 'w',
                if_sheet_exists='overlay'
            ) as writer:
                df.to_excel(writer, sheet_name='对比结果', index=False, header=not Path(self.comparison_table).exists())
                
                # 设置样式
                worksheet = writer.sheets['对比结果']
                
                # 设置列宽
                worksheet.column_dimensions['A'].width = 30  # 用户输入列
                for col in range(1, len(self.test_configs)+1):
                    col_letter = chr(ord('B') + col - 1)
                    worksheet.column_dimensions[col_letter].width = 50
                    
                # 设置单元格样式
                for row in worksheet.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(vertical='top', wrap_text=True)
                        if cell.row == 1:  # 表头行
                            cell.font = Font(bold=True)
                        elif cell.value and cell.value.startswith('→'):
                            cell.font = Font(color='666666')  # 灰色

            self._logger.info(f"保存结果到: {self.comparison_table}")

        except Exception as e:
            self._logger.error(f"保存结果失败: {str(e)}")
            raise

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

    async def stream_chat(self, session: aiohttp.ClientSession, message: str, config: TestConfig) -> Dict[str, Any]:
        """调用流式对话 API"""
        url = f"{self.base_url}/api/{self.api_version}/chat/{self.agent_id}/stream"
        try:
            request_data = {
                "message": message,
                "user_id": f"test_user_{config.name}",
                "remark": f"配置测试: {config.name}",
                "config": config.to_dict()
            }
            
            response_text = ""
            metadata = None
            
            async with session.post(url, json=request_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API错误: {response.status} - {error_text}")
                
                # 读取SSE流
                async for chunk in response.content:
                    chunk_text = chunk.decode('utf-8').strip()
                    if not chunk_text:
                        continue
                        
                    # 解析SSE格式
                    for line in chunk_text.split('\n'):
                        if line.startswith('event: metadata'):
                            continue
                        if line.startswith('event: error'):
                            continue
                        if line.startswith('data: '):
                            data = line[6:]  # 去掉 "data: " 前缀
                            try:
                                # 尝试解析为JSON（可能是元数据）
                                data_json = json.loads(data)
                                if isinstance(data_json, dict):
                                    metadata = data_json
                                else:
                                    print(data, end='', flush=True)
                                    response_text += data
                            except json.JSONDecodeError:
                                # 普通文本内容
                                print(data, end='', flush=True)
                                response_text += data
                
                print("\n---")  # 换行分隔符
                print()  # 空行
                
                return {
                    "response": response_text.strip(),  # 直接保存纯文本内容
                    "metadata": metadata.get('processed_entity_memory', '') if metadata else ''
                }
                            
        except Exception as e:
            self._logger.error(f"聊天请求失败: {str(e)}")
            return {
                "response": "",
                "metadata": "",
                "error": str(e)
            }

    async def run_comparison_tests(self):
        """运行对比测试"""
        self._logger.info("开始执行对比测试...")
        test_cases = await self.load_test_cases()
        
        async with aiohttp.ClientSession() as session:
            # 按批次处理测试用例
            batch_size = 3  # 每批处理3个测试用例
            for i in range(0, len(test_cases), batch_size):
                batch = test_cases[i:i+batch_size]
                self._logger.info(f"\n=== 开始处理第 {i//batch_size + 1} 批测试用例 ===")
                
                # 处理这一批次的所有测试用例
                for test_case in batch:
                    self._logger.info(f"\n--- 测试用例: {test_case['topic']} ---")
                    
                    # 为每个配置创建一个测试任务
                    for config in self.test_configs:
                        async with self.semaphore:  # 使用信号量控制并发
                            self._logger.info(f"\n配置: {config.name}")
                            print(f"\n用户: {test_case['input']}")
                            print(f"[{config.name}] 回复: ", end='', flush=True)
                            
                            result = await self._execute_test(session, test_case, config)
                            self.results.append(result)
                    
                    # 每个测试用例完成后保存结果
                    self.save_comparison_results()
                    
                    # 测试用例之间添加短暂延迟
                    await asyncio.sleep(1)
                
                # 批次之间添加较长延迟
                self._logger.info(f"\n=== 第 {i//batch_size + 1} 批测试用例完成 ===")
                await asyncio.sleep(5)

            self._logger.info("\n=== 对比测试完成 ===")

    async def _execute_test(self, session, test_case, config):
        """执行单个配置的测试"""
        try:
            result = await self.stream_chat(session, test_case['input'], config)
            return {
                'test_id': f"{test_case['test_type']}_{hash(test_case['input'])}",
                'topic': test_case['topic'],
                'input': test_case['input'],
                'config_name': config.name,
                'response': result["response"],  # 纯文本内容
                'metadata': result["metadata"],  # 处理后的实体记忆
                'error': '',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self._logger.error(f"测试执行失败: {str(e)}")
            return {
                'test_id': f"{test_case['test_type']}_{hash(test_case['input'])}",
                'topic': test_case['topic'],
                'input': test_case['input'],
                'config_name': config.name,
                'response': '',
                'metadata': '',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

async def main():
    """主函数"""
    # 调整并行参数
    tester = ComparisonTester()
    tester.semaphore = asyncio.Semaphore(3)  # 设置最大并行数
    await tester.run_comparison_tests()

if __name__ == "__main__":
    asyncio.run(main()) 