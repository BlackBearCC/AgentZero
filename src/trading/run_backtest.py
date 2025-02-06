import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import argparse


# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
from src.utils.logger import Logger

# 创建回测专用logger
logger = Logger("backtest")
if project_root not in sys.path:
    sys.path.append(project_root)
    logger.info(f"添加 {project_root} 到 Python 路径")

try:
    from src.trading.engine.backtest import BacktestRunner
    from src.trading.strategies.grid import AutoGridStrategy
except ImportError as e:
    logger.error(f"导入错误: {str(e)}")
    logger.error(f"当前 Python 路径: {sys.path}")
    raise

def parse_date(date_str):
    """解析日期字符串为datetime对象"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"无效的日期格式: {date_str}. 请使用 YYYY-MM-DD 格式")

async def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='网格交易策略回测')
    parser.add_argument('--start', type=parse_date, default="2024-01-01",
                      help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=parse_date, default="2025-02-05",
                      help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='DOGE/USDT',
                      help='交易对 (默认: DOGE/USDT)')
    parser.add_argument('--timeframe', type=str, default='15m',
                      help='K线周期 (默认: 15m)')
    parser.add_argument('--cash', type=float, default=10000,
                      help='初始资金 (默认: 10000)')
    
    args = parser.parse_args()
    
    # 检查日期范围
    if args.end <= args.start:
        logger.error("结束日期必须晚于开始日期")
        return
        
    logger.info("开始执行回测...")
    logger.info(f"回测时间范围: {args.start.date()} 到 {args.end.date()}")
    logger.info(f"交易对: {args.symbol}")
    logger.info(f"K线周期: {args.timeframe}")
    logger.info(f"初始资金: ${args.cash:.2f}")
    
    # 回测参数配置
    params = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'start': args.start,
        'end': args.end,
        'initial_cash': args.cash,
        'commission': 0.001,
        'strategy_params': {
            'grid_number': 50,         # 网格数量
            'position_size': 0.02,      # 每格仓位
            'atr_period': 14,          # ATR周期
            'vol_period': 20,          # 波动率周期
            'grid_min_spread': 0.002,  # 最小网格间距
            'grid_max_spread': 0.06,   # 最大网格间距
            'grid_expansion': 2.0      # 网格区间扩展系数
        }
    }

    # 创建并运行回测
    runner = BacktestRunner()
    try:
        logger.info("初始化回测引擎...")
        # 添加数据加载日志
        logger.info("开始加载历史数据...")
        result = await runner.run(**params)
        
        # 打印回测结果
        logger.info("\n=== 回测结果 ===")
        logger.info(f"初始资金: ${args.cash:.2f}")
        logger.info(f"总收益率: {((result['final_value']/args.cash - 1) * 100):.2f}%")
        logger.info(f"交易次数: {result.get('trade_count', 0)}")
        logger.info(f"回测周期: {args.start.date()} 到 {args.end.date()}")
        
    except Exception as e:
        logger.error(f"回测执行出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    # 运行异步主函数
    logger.info("启动回测脚本...")
    asyncio.run(main())
    logger.info("回测完成") 