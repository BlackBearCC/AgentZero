import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to Python path")

try:
    from src.trading.engine.backtest import BacktestRunner
    from src.trading.strategies.grid import AutoGridStrategy
except ImportError as e:
    print(f"导入错误: {str(e)}")
    print(f"当前 Python 路径: {sys.path}")
    raise

async def main():
    logger.info("开始执行回测...")
    
    # 设置回测时间范围（使用历史数据）
    end_date = datetime.now()- timedelta(days=2)
    start_date = end_date - timedelta(days=90)  # 最近30天数据
    
    # 回测参数配置
    params = {
        'symbol': 'DOGE/USDT',
        'timeframe': '15m',
        'start': start_date,
        'end': end_date,
        'initial_cash': 10000,
        'commission': 0.001,
        # 'strategy_params': {
        #     'base_spacing': 0.02,      # 基础网格间距3%
        #     'dynamic_ratio': 1.5,      # 动态扩展系数
        #     'leverage': 20,            # 杠杆倍数
        #     'max_grids': 100,           # 网格数量
        #     'rebalance_bars': 1000,      # 3根K线重新平衡
        #     'max_drawdown': 20,       # 最大回撤限制
        #     'atr_period': 14,         # ATR周期
        #     'ema_fast': 20,           # 快速EMA
        #     'ema_slow': 50            # 慢速EMA
        # }
    }

    logger.info(f"回测参数: {params}")

    # 创建并运行回测
    runner = BacktestRunner()
    try:
        logger.info("初始化回测引擎...")
        await runner.run(**params)
    except Exception as e:
        logger.error(f"回测执行出错: {str(e)}", exc_info=True)
        raise  # 添加这行以显示完整错误信息

if __name__ == '__main__':
    # 运行异步主函数
    logger.info("启动回测脚本...")
    asyncio.run(main())
    logger.info("回测完成") 