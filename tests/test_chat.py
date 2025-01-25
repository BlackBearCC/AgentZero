from typing import List, Dict, Any
import asyncio
import csv
import json
import aiohttp
from datetime import datetime
from pathlib import Path
import logging

class ChatTester:
    def __init__(self, remark: str = ''):
        """初始化测试配置
        
        Args:
            remark: 测试备注信息，将用于所有测试用例
        """
        self.base_url = "http://localhost:8000"
        self.api_version = "v1"
        self.agent_id = "qiyu_001"
        self.remark = remark  # 存储备注信息
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self._logger = logging.getLogger(__name__)
        
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
        
    async def save_test_results(self, results: List[Dict[str, Any]]):
        """保存测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('tests/results')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f'chat_test_results_{timestamp}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            self._logger.info(f"测试结果已保存到: {output_file}")

    async def stream_chat(self, session: aiohttp.ClientSession, message: str) -> str:
        """调用流式对话 API"""
        url = f"{self.base_url}/api/{self.api_version}/chat/{self.agent_id}/stream"
        try:
            # 添加备注到请求体
            async with session.post(url, json={
                "message": message,
                "remark": self.remark
            }) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                # 读取流式响应
                response_text = ""
                async for chunk in response.content.iter_chunks():
                    chunk_data, _ = chunk
                    chunk_text = chunk_data.decode('utf-8')
                    print(chunk_text, end='', flush=True)
                    response_text += chunk_text
                
                print("\n---")  # 换行分隔符
                print()  # 空行
                
                return response_text
                            
        except Exception as e:
            self._logger.error(f"Stream chat error: {str(e)}")
            raise

    async def test_multi_turn_chat(self) -> List[Dict[str, Any]]:
        """执行多轮对话测试"""
        async with aiohttp.ClientSession() as session:
            test_cases = await self.load_test_cases()
            results = []
            
            current_topic = None
            
            for case in test_cases:
                # 如果话题改变,重置对话历史
                if case['topic'] != current_topic:
                    current_topic = case['topic']
                    self._logger.info(f"\n=== 开始新话题: {current_topic} ===")
                
                # 记录测试信息
                test_info = {
                    'test_type': case['test_type'],
                    'topic': case['topic'],
                    'input': case['input'],
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
                    # 调用 API 生成回复
                    self._logger.info(f"\n用户: {case['input']}")
                    response = await self.stream_chat(session, case['input'])
                    
                    # 记录结果
                    test_info.update({
                        'response': response,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    self._logger.error(f"Error in test case: {str(e)}")
                    test_info.update({
                        'status': 'error',
                        'error': str(e)
                    })
                    
                results.append(test_info)
                
                # 每轮对话后短暂暂停
                await asyncio.sleep(1)
            
            # 保存测试结果
            await self.save_test_results(results)
            return results

    async def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析测试结果"""
        analysis = {
            'total_cases': len(results),
            'success_count': 0,
            'error_count': 0,
            'topics': set(),
            'average_response_time': 0,
            'error_cases': []
        }
        
        total_time = 0
        
        for result in results:
            analysis['topics'].add(result['topic'])
            
            if result['status'] == 'success':
                analysis['success_count'] += 1
                if 'timestamp' in result:
                    response_time = datetime.fromisoformat(result['timestamp'])
                    total_time += response_time.microsecond
            else:
                analysis['error_count'] += 1
                analysis['error_cases'].append({
                    'topic': result['topic'],
                    'input': result['input'],
                    'error': result['error']
                })
        
        if analysis['success_count'] > 0:
            analysis['average_response_time'] = total_time / analysis['success_count']
        analysis['topics'] = list(analysis['topics'])
        
        return analysis

    async def run_tests(self):
        """运行所有测试"""
        self._logger.info("开始执行测试...")
        
        try:
            # 执行多轮对话测试
            results = await self.test_multi_turn_chat()
            
            # 分析结果
            analysis = await self.analyze_results(results)
            
            # 保存分析报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path('tests/results')
            analysis_file = output_dir / f'analysis_report_{timestamp}.json'
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
                self._logger.info(f"分析报告已保存到: {analysis_file}")
            
            # 输出测试总结
            self._logger.info("\n=== 测试总结 ===")
            self._logger.info(f"总用例数: {analysis['total_cases']}")
            self._logger.info(f"成功数: {analysis['success_count']}")
            self._logger.info(f"失败数: {analysis['error_count']}")
            self._logger.info(f"测试话题: {', '.join(analysis['topics'])}")
            self._logger.info(f"平均响应时间: {analysis['average_response_time']}ms")
            
            if analysis['error_cases']:
                self._logger.info("\n=== 错误用例 ===")
                for case in analysis['error_cases']:
                    self._logger.error(f"话题: {case['topic']}")
                    self._logger.error(f"输入: {case['input']}")
                    self._logger.error(f"错误: {case['error']}\n")
                    
        except Exception as e:
            self._logger.error(f"测试执行失败: {str(e)}")
            raise

async def main():
    """主函数"""
    remark = "单轮召回记忆队列，无概要"
    
    tester = ChatTester(remark=remark)
    await tester.run_tests()

if __name__ == "__main__":
    asyncio.run(main()) 