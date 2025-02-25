from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
import torch
from pathlib import Path
import time
from joblib import Parallel, delayed
import os
import json
from tqdm import tqdm

from src.trading.feeds.crypto_feed import DataManager
from src.trading.models.datasets import MarketMicrostructureDataset
from src.trading.models.transformer import AlphaTransformer

class TransformerBacktester:
    """Transformer模型回测器 - 性能优化版"""
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.load_model(model_path)
        
    def load_model(self, path: str):
        """加载预训练模型"""
        # 使用map_location避免GPU内存泄漏
        checkpoint = torch.load(path, map_location=self.device)
        self.model = AlphaTransformer(**checkpoint['model_config']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler_state']
        self.model.eval()  # 确保模型处于评估模式
        print(f"模型加载成功 (使用{self.device})")
        
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    initial_capital: float = 1_000_000,
                    transaction_cost: float = 0.0004,
                    position_limit: float = 0.5,      # 最大仓位
                    volatility_threshold: float = 0.015,  # 波动率阈值
                    signal_threshold: float = 0.3,    # 信号阈值
                    smoothing_factor: float = 0.7,    # 平滑因子
                    prediction_horizon: int = 14,
                    verbose: bool = False,  # 默认关闭详细输出以提高速度
                    dynamic_position_sizing: bool = True,  # 动态仓位控制
                    stop_loss_pct: float = 0.02,      # 止损百分比
                    take_profit_pct: float = 0.05,    # 止盈百分比
                    ) -> Dict:
        """运行回测 - 性能优化版"""
        if verbose:
            print("开始回测...")
            start_time = time.time()
        
        # 准备数据 - 预先处理以提高性能
        dataset = MarketMicrostructureDataset(data)
        features = dataset.features
        
        # 初始化变量 - 使用NumPy数组而非列表以提高性能
        window_size = 60
        data_length = len(features) - prediction_horizon
        positions = np.zeros(data_length - window_size)
        equity = np.zeros(data_length - window_size + 1)
        equity[0] = initial_capital
        
        current_position = 0
        entry_price = 0
        trades = []
        signal_records = []
        
        # 风险管理变量
        stop_loss_price = 0
        take_profit_price = 0
        
        # 批量预测 - 减少GPU调用次数
        batch_size = 32  # 可调整
        
        # 遍历每个时间点
        for i in range(window_size, len(features) - prediction_horizon, batch_size):
            # 确定当前批次的结束索引
            end_idx = min(i + batch_size, len(features) - prediction_horizon)
            batch_indices = range(i, end_idx)
            
            # 批量准备输入数据
            batch_inputs = []
            for idx in batch_indices:
                window = features[idx-window_size:idx]
                batch_inputs.append(window)
            
            # 转换为张量并进行批量预测
            batch_tensor = torch.FloatTensor(np.array(batch_inputs)).to(self.device)
            with torch.no_grad():
                batch_preds = self.model(batch_tensor)
            
            # 处理每个预测结果
            for batch_idx, idx in enumerate(batch_indices):
                current_time = data.index[idx]
                current_price = data['close'].iloc[idx]
                
                # 获取当前预测
                future_returns = batch_preds[batch_idx].cpu().numpy()
                
                # 信号生成
                short_term = np.mean(future_returns[0:3])     # 短期信号
                medium_term = np.mean(future_returns[3:8])    # 中期信号
                long_term = np.mean(future_returns[8:14])     # 长期信号
                
                # 计算波动率 - 使用EWMA波动率估计
                volatility = np.std(future_returns)
                if idx > window_size:
                    # 结合历史波动率和当前波动率
                    volatility = 0.94 * volatility + 0.06 * data['close'].iloc[idx-20:idx].pct_change().std()
                
                # 记录信号
                signal_records.append({
                    'time': current_time,
                    'price': current_price,
                    'short_term': short_term,
                    'medium_term': medium_term,
                    'long_term': long_term,
                    'volatility': volatility
                })
                
                # 信号平滑处理
                if len(signal_records) > 0:
                    last_medium_term = signal_records[-1].get('medium_term', medium_term)
                    medium_term = smoothing_factor * last_medium_term + \
                                 (1 - smoothing_factor) * medium_term
                
                # 止损和止盈检查
                target_position = current_position  # 默认保持当前仓位
                
                if current_position > 0 and stop_loss_price > 0 and current_price <= stop_loss_price:
                    # 触发止损
                    target_position = 0
                    trades.append({
                        'time': current_time,
                        'type': 'stop_loss',
                        'price': current_price,
                        'size': -current_position,
                        'cost': abs(current_position) * transaction_cost * equity[idx-window_size],
                        'pnl': (current_price/entry_price - 1) * current_position * equity[idx-window_size],
                        'short_term': short_term,
                        'medium_term': medium_term,
                        'long_term': long_term
                    })
                    current_position = 0
                    stop_loss_price = 0
                    take_profit_price = 0
                elif current_position < 0 and stop_loss_price > 0 and current_price >= stop_loss_price:
                    # 触发止损
                    target_position = 0
                    trades.append({
                        'time': current_time,
                        'type': 'stop_loss',
                        'price': current_price,
                        'size': -current_position,
                        'cost': abs(current_position) * transaction_cost * equity[idx-window_size],
                        'pnl': (1 - current_price/entry_price) * abs(current_position) * equity[idx-window_size],
                        'short_term': short_term,
                        'medium_term': medium_term,
                        'long_term': long_term
                    })
                    current_position = 0
                    stop_loss_price = 0
                    take_profit_price = 0
                elif current_position > 0 and take_profit_price > 0 and current_price >= take_profit_price:
                    # 触发止盈
                    target_position = 0
                    trades.append({
                        'time': current_time,
                        'type': 'take_profit',
                        'price': current_price,
                        'size': -current_position,
                        'cost': abs(current_position) * transaction_cost * equity[idx-window_size],
                        'pnl': (current_price/entry_price - 1) * current_position * equity[idx-window_size],
                        'short_term': short_term,
                        'medium_term': medium_term,
                        'long_term': long_term
                    })
                    current_position = 0
                    stop_loss_price = 0
                    take_profit_price = 0
                elif current_position < 0 and take_profit_price > 0 and current_price <= take_profit_price:
                    # 触发止盈
                    target_position = 0
                    trades.append({
                        'time': current_time,
                        'type': 'take_profit',
                        'price': current_price,
                        'size': -current_position,
                        'cost': abs(current_position) * transaction_cost * equity[idx-window_size],
                        'pnl': (1 - current_price/entry_price) * abs(current_position) * equity[idx-window_size],
                        'short_term': short_term,
                        'medium_term': medium_term,
                        'long_term': long_term
                    })
                    current_position = 0
                    stop_loss_price = 0
                    take_profit_price = 0
                else:
                    # 新的多空切换逻辑
                    # 信号一致性检查
                    signals_aligned = (
                        np.sign(short_term) == np.sign(medium_term) and 
                        np.sign(medium_term) == np.sign(long_term)
                    )
                    
                    # 信号强度检查
                    signal_strong = abs(medium_term) > volatility * signal_threshold
                    
                    if signals_aligned and signal_strong:
                        if medium_term > 0:  # 看多信号
                            if current_position <= 0:  # 当前空仓或空头
                                # 平空仓并做多
                                if dynamic_position_sizing:
                                    # 动态仓位大小基于信号强度和波动率
                                    signal_strength = medium_term / volatility
                                    target_position = min(position_limit * signal_strength / 2, position_limit)
                                else:
                                    target_position = position_limit
                        else:  # 看空信号
                            if current_position >= 0:  # 当前空仓或多头
                                # 平多仓并做空
                                if dynamic_position_sizing:
                                    # 动态仓位大小基于信号强度和波动率
                                    signal_strength = abs(medium_term) / volatility
                                    target_position = max(-position_limit * signal_strength / 2, -position_limit)
                                else:
                                    target_position = -position_limit
                
                # 执行交易
                position_change = target_position - current_position
                
                if abs(position_change) > 0.05:  # 最小交易阈值
                    # 计算交易成本和PNL
                    transaction_fee = abs(position_change) * transaction_cost * equity[idx-window_size]
                    
                    # 计算当前交易的PNL
                    if current_position != 0:  # 如果是平仓或调仓
                        pnl = (current_price/entry_price - 1) * current_position * equity[idx-window_size]
                    else:
                        pnl = 0
                    
                    trade_type = 'long' if target_position > 0 else 'short' if target_position < 0 else 'close'
                    trades.append({
                        'time': current_time,
                        'type': trade_type,
                        'price': current_price,
                        'size': position_change,
                        'cost': transaction_fee,
                        'pnl': pnl,
                        'short_term': short_term,
                        'medium_term': medium_term,
                        'long_term': long_term
                    })
                    
                    current_position = target_position
                    if current_position != 0:
                        entry_price = current_price
                        # 设置止损和止盈价格
                        if current_position > 0:  # 多头
                            stop_loss_price = current_price * (1 - stop_loss_pct)
                            take_profit_price = current_price * (1 + take_profit_pct)
                        else:  # 空头
                            stop_loss_price = current_price * (1 + stop_loss_pct)
                            take_profit_price = current_price * (1 - take_profit_pct)
                    else:
                        stop_loss_price = 0
                        take_profit_price = 0
                
                positions[idx-window_size] = current_position
                
                # 更新权益
                if current_position != 0:
                    price_change = data['close'].iloc[idx] / data['close'].iloc[idx-1] - 1
                    equity_change = price_change * current_position * equity[idx-window_size]
                    
                    # 扣除交易成本
                    if len(trades) > 0 and trades[-1]['time'] == current_time:
                        equity_change -= trades[-1]['cost']
                        
                    equity[idx-window_size+1] = equity[idx-window_size] + equity_change
                else:
                    equity[idx-window_size+1] = equity[idx-window_size]
                
                # 打印详细信息
                if verbose and idx % 500 == 0:
                    # 计算当前收益率和最大回撤
                    current_return = (equity[idx-window_size+1] / initial_capital - 1) * 100
                    equity_series_temp = pd.Series(equity[:idx-window_size+2])
                    max_drawdown = ((equity_series_temp.cummax() - equity_series_temp) / equity_series_temp.cummax()).max() * 100
                    
                    print(f"""
======== 回测状态更新 ========
时间: {current_time}
价格: {current_price:.2f}
账户资金: {equity[idx-window_size+1]:,.2f} USDT
当前收益率: {current_return:.2f}%
最大回撤: {max_drawdown:.2f}%
当前仓位: {current_position:.2f}
交易次数: {len(trades)}
==============================
                    """)
        
        # 在回测循环结束后，创建返回数据结构
        dates = data.index[window_size:window_size+len(equity)]
        equity_series = pd.Series(equity, index=dates)
        position_series = pd.Series(positions, index=dates[:-1])  # positions比equity少一个
        
        # 转换交易记录为DataFrame
        if trades:  # 如果有交易
            trade_df = pd.DataFrame(trades)
            trade_df['time'] = pd.to_datetime(trade_df['time'])
        else:  # 如果没有交易
            trade_df = pd.DataFrame(columns=['time', 'type', 'price', 'size', 'pnl', 'cost'])
        
        # 转换信号记录为DataFrame
        signal_df = pd.DataFrame(signal_records)
        
        # 计算回测指标
        metrics = self._calculate_metrics(equity, positions, trades, dates)
        
        if verbose:
            end_time = time.time()
            print(f"回测完成，耗时: {end_time - start_time:.2f}秒")
            print(f"总收益率: {metrics['total_return']:.2f}%")
            print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
            print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
            print(f"夏普比率: {metrics['sharpe']:.2f}")
        
        return {
            'equity': equity_series,
            'positions': position_series,
            'trades': trade_df,
            'signals': signal_df,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, equity, positions, trades, dates) -> Dict:
        """计算回测指标"""
        # 计算总收益率
        total_return = (equity[-1] / equity[0] - 1) * 100
        
        # 计算年化收益率
        days = (dates[-1] - dates[0]).days
        annual_return = (equity[-1] / equity[0]) ** (365 / max(days, 1)) - 1
        
        # 计算最大回撤
        equity_series = pd.Series(equity)
        drawdown = (equity_series.cummax() - equity_series) / equity_series.cummax() * 100
        max_drawdown = drawdown.max()
        
        # 计算年化波动率
        daily_returns = pd.Series(equity).pct_change().dropna()
        annual_vol = daily_returns.std() * np.sqrt(365)
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # 计算交易统计
        if trades:
            trade_df = pd.DataFrame(trades)
            winning_trades = trade_df[trade_df['pnl'] > 0]
            losing_trades = trade_df[trade_df['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trade_df) if len(trade_df) > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
            
            avg_trade_return = trade_df['pnl'].mean() if len(trade_df) > 0 else 0
            total_cost = trade_df['cost'].sum()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_return = 0
            total_cost = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'total_cost': total_cost,
            'trade_count': len(trades),
            'annual_vol': annual_vol
        }

class FastHyperparameterOptimizer:
    """高性能超参数优化器"""
    def __init__(self, model_path: str, data: pd.DataFrame):
        self.model_path = model_path
        self.data = data
        self.backtester = TransformerBacktester(model_path)
    
    def bayesian_optimization(self, 
                             param_ranges: Dict[str, Tuple[float, float]], 
                             n_trials: int = 50,
                             metric: str = 'sharpe') -> Dict:
        """
        贝叶斯优化 - 比网格搜索更高效
        
        参数:
            param_ranges: 参数范围，如 {'position_limit': (0.3, 0.7), 'signal_threshold': (0.2, 0.4)}
            n_trials: 优化尝试次数
            metric: 优化指标
            
        返回:
            最优参数和结果
        """
        try:
            import optuna
        except ImportError:
            print("请先安装optuna: pip install optuna")
            return {}
        
        # 创建优化研究
        study_name = f"trading_optimization_{int(time.time())}"
        storage_name = f"sqlite:///{study_name}.db"
        
        # 定义目标函数
        def objective(trial):
            # 从参数范围中采样
            params = {}
            for param_name, (low, high) in param_ranges.items():
                params[param_name] = trial.suggest_float(param_name, low, high)
            
            # 运行回测
            try:
                results = self.backtester.run_backtest(
                    data=self.data,
                    verbose=False,
                    **params
                )
                
                metrics = results['metrics']
                
                # 根据优化指标返回值
                if metric == 'sharpe':
                    return metrics['sharpe']
                elif metric == 'return':
                    return metrics['annual_return']
                elif metric == 'drawdown':
                    return -metrics['max_drawdown']  # 负号使得最小化最大回撤
                elif metric == 'calmar':
                    return metrics['annual_return'] / (metrics['max_drawdown']/100)
                else:
                    return metrics['sharpe']
            except Exception as e:
                print(f"参数评估失败: {params}, 错误: {e}")
                return float('-inf')  # 失败时返回负无穷
        
        # 创建研究
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            load_if_exists=True
        )
        
        # 运行优化
        print(f"开始贝叶斯优化，计划{n_trials}次尝试...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 获取最佳参数
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n优化完成！最佳{metric}值: {best_value:.4f}")
        print("最优参数组合:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.4f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f'bayesian_opt_results_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_value': best_value,
                'metric': metric,
                'n_trials': n_trials,
                'timestamp': timestamp
            }, f, indent=4)
        
        print(f"结果已保存到 {result_file}")
        
        # 使用最优参数运行一次回测以获取详细结果
        final_results = self.backtester.run_backtest(
            data=self.data,
            verbose=True,
            **best_params
        )
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'final_results': final_results
        }
    
    def walk_forward_bayesian(self,
                             param_ranges: Dict[str, Tuple[float, float]],
                             window_size: int = 30,  # 窗口大小(天)
                             step_size: int = 15,    # 步长(天)
                             n_trials: int = 30,     # 每个窗口的优化尝试次数
                             metric: str = 'sharpe') -> Dict:
        """
        前推式贝叶斯优化 - 量化机构标准方法
        
        参数:
            param_ranges: 参数范围
            window_size: 训练窗口大小(天)
            step_size: 前进步长(天)
            n_trials: 每个窗口的优化尝试次数
            metric: 优化指标
            
        返回:
            前推式优化结果
        """
        try:
            import optuna
        except ImportError:
            print("请先安装optuna: pip install optuna")
            return {}
        
        # 将数据按日期分割成多个窗口
        dates = self.data.index
        start_date = dates[0]
        end_date = dates[-1]
        
        # 生成时间窗口
        windows = []
        current_start = start_date
        while current_start < end_date:
            train_end = current_start + pd.Timedelta(days=window_size)
            test_end = train_end + pd.Timedelta(days=step_size)
            
            if test_end > end_date:
                test_end = end_date
                
            if train_end > end_date:
                break
                
            windows.append((current_start, train_end, test_end))
            current_start = current_start + pd.Timedelta(days=step_size)
        
        print(f"前推式贝叶斯优化: 共{len(windows)}个时间窗口")
        
        # 存储每个窗口的最优参数和测试结果
        window_results = []
        all_test_equity = pd.Series(dtype=float)
        
        # 创建进度记录文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_file = f"wf_bayesian_progress_{timestamp}.csv"
        
        with open(progress_file, 'w') as f:
            header = ['window', 'train_start', 'train_end', 'test_end']
            header.extend(list(param_ranges.keys()))
            header.extend(['sharpe', 'return', 'drawdown', 'win_rate'])
            f.write(','.join(header) + '\n')
        
        # 对每个窗口进行优化
        for i, (train_start, train_end, test_end) in enumerate(windows):
            print(f"\n窗口 {i+1}/{len(windows)}: 训练 {train_start.date()} 到 {train_end.date()}, 测试到 {test_end.date()}")
            
            # 提取训练数据
            train_data = self.data[(self.data.index >= train_start) & (self.data.index < train_end)]
            
            # 提取测试数据
            test_data = self.data[(self.data.index >= train_end) & (self.data.index < test_end)]
            
            if len(train_data) < 100 or len(test_data) < 20:  # 确保有足够的数据
                print(f"  跳过窗口 {i+1}: 数据不足")
                continue
            
            # 创建优化研究
            study_name = f"wf_optimization_{i}_{int(time.time())}"
            storage_name = f"sqlite:///{study_name}.db"
            
            # 定义目标函数
            def objective(trial):
                # 从参数范围中采样
                params = {}
                for param_name, (low, high) in param_ranges.items():
                    params[param_name] = trial.suggest_float(param_name, low, high)
                
                # 在训练数据上运行回测
                try:
                    results = self.backtester.run_backtest(
                        data=train_data,
                        verbose=False,
                        **params
                    )
                    
                    metrics = results['metrics']
                    
                    # 根据优化指标返回值
                    if metric == 'sharpe':
                        return metrics['sharpe']
                    elif metric == 'return':
                        return metrics['annual_return']
                    elif metric == 'drawdown':
                        return -metrics['max_drawdown']
                    elif metric == 'calmar':
                        return metrics['annual_return'] / (metrics['max_drawdown']/100)
                    else:
                        return metrics['sharpe']
                except Exception as e:
                    print(f"  参数评估失败: {params}, 错误: {e}")
                    return float('-inf')
            
            # 创建研究
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                direction="maximize",
                load_if_exists=True
            )
            
            # 运行优化
            print(f"  开始窗口 {i+1} 贝叶斯优化，计划{n_trials}次尝试...")
            study.optimize(objective, n_trials=n_trials)
            
            # 获取最佳参数
            best_params = study.best_params
            best_value = study.best_value
            
            print(f"  窗口 {i+1} 优化完成！最佳{metric}值: {best_value:.4f}")
            print("  最优参数组合:")
            for param, value in best_params.items():
                print(f"    {param}: {value:.4f}")
            
            # 在测试数据上验证
            test_results = self.backtester.run_backtest(
                data=test_data,
                verbose=False,
                **best_params
            )
            
            test_metrics = test_results['metrics']
            test_equity = test_results['equity']
            
            # 记录结果
            result = {
                'window': i+1,
                'train_start': train_start,
                'train_end': train_end,
                'test_end': test_end,
                'train_sharpe': best_value,
                'test_sharpe': test_metrics['sharpe'],
                'test_return': test_metrics['annual_return'],
                'test_drawdown': test_metrics['max_drawdown'],
                'test_win_rate': test_metrics['win_rate'],
                'equity': test_equity
            }
            
            # 添加最优参数
            for param, value in best_params.items():
                result[param] = value
                
            window_results.append(result)
            
            # 合并测试期权益曲线
            if isinstance(test_equity, pd.Series):
                all_test_equity = pd.concat([all_test_equity, test_equity])
            
            # 保存进度
            with open(progress_file, 'a') as f:
                values = [str(i+1), str(train_start.date()), str(train_end.date()), str(test_end.date())]
                for param_name in param_ranges.keys():
                    values.append(str(best_params.get(param_name, '')))
                values.extend([
                    str(best_value),
                    str(test_metrics['annual_return']),
                    str(test_metrics['max_drawdown']),
                    str(test_metrics['win_rate'])
                ])
                f.write(','.join(values) + '\n')
            
            print(f"  测试期表现: 夏普比率={test_metrics['sharpe']:.2f}, 收益率={test_metrics['annual_return']*100:.2f}%, 最大回撤={test_metrics['max_drawdown']:.2f}%")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(window_results)
        
        # 计算参数稳定性
        param_stability = {}
        for param in param_ranges.keys():
            if param in results_df.columns:
                param_stability[param] = {
                    'mean': results_df[param].mean(),
                    'std': results_df[param].std(),
                    'cv': results_df[param].std() / results_df[param].mean() if results_df[param].mean() != 0 else float('inf')
                }
        
        # 计算平均最优参数
        avg_params = {}
        for param in param_ranges.keys():
            if param in results_df.columns:
                avg_params[param] = results_df[param].mean()
        
        # 保存结果
        results_file = f'wf_bayesian_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        # 保存平均最优参数
        params_file = f'wf_bayesian_params_{timestamp}.json'
        with open(params_file, 'w') as f:
            json.dump({
                'avg_params': avg_params,
                'param_stability': param_stability,
                'timestamp': timestamp
            }, f, indent=4)
        
        print(f"\n前推式贝叶斯优化完成！")
        print(f"结果已保存到 {results_file}")
        print(f"平均最优参数已保存到 {params_file}")
        
        # 使用平均最优参数运行一次完整回测
        print("\n使用平均最优参数运行完整回测...")
        final_results = self.backtester.run_backtest(
            data=self.data,
            verbose=True,
            **avg_params
        )
        
        # 绘制参数稳定性图表
        plt.figure(figsize=(12, 6))
        for param in param_ranges.keys():
            if param in results_df.columns:
                plt.plot(results_df['window'], results_df[param], 'o-', label=param)
        
        plt.title('参数稳定性分析')
        plt.xlabel('窗口')
        plt.ylabel('参数值')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'param_stability_{timestamp}.png')
        
        # 绘制测试期权益曲线
        if not all_test_equity.empty:
            plt.figure(figsize=(12, 6))
            all_test_equity.plot()
            plt.title('前推式优化测试期权益曲线')
            plt.xlabel('日期')
            plt.ylabel('权益')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'wf_test_equity_{timestamp}.png')
        
        return {
            'window_results': results_df,
            'avg_params': avg_params,
            'param_stability': param_stability,
            'final_results': final_results
        }

def run_optimization_comparison(data_path: str = None):
    """运行优化方法对比"""
    # 配置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 加载回测数据
    if data_path and os.path.exists(data_path):
        data = pd.read_pickle(data_path)
    else:
        data_mgr = DataManager()
        data = data_mgr.get_feed(
            symbol='BTC/USDT',
            timeframe='15m',
            start=datetime(2024, 1, 1),
            end=datetime(2024, 5, 24)
        )['indicator']
        # 保存数据以便重用
        data.to_pickle('backtest_data.pkl')
    
    # 创建回测器
    model_path = 'models/alpha_v1.pth'
    backtester = TransformerBacktester(model_path)
    
    # 默认参数回测
    print("运行默认参数回测...")
    default_results = backtester.run_backtest(
        data=data,
        initial_capital=1_000_000,
        transaction_cost=0.0004,
        position_limit=0.5,
        volatility_threshold=0.015,
        signal_threshold=0.3,
        smoothing_factor=0.7,
        prediction_horizon=14,
        verbose=True
    )
    
    # 创建优化器
    optimizer = FastHyperparameterOptimizer(model_path, data)
    
    # 定义参数范围
    param_ranges = {
        'position_limit': (0.3, 0.7),
        'signal_threshold': (0.2, 0.4),
        'volatility_threshold': (0.01, 0.02),
        'smoothing_factor': (0.5, 0.9),
        'stop_loss_pct': (0.01, 0.03),
        'take_profit_pct': (0.03, 0.07)
    }
    
    # 运行标准贝叶斯优化
    print("\n\n运行标准贝叶斯优化...")
    bayes_results = optimizer.bayesian_optimization(
        param_ranges=param_ranges,
        n_trials=50,
        metric='sharpe'
    )
    
    # 运行前推式贝叶斯优化
    print("\n\n运行前推式贝叶斯优化...")
    wf_bayes_results = optimizer.walk_forward_bayesian(
        param_ranges=param_ranges,
        window_size=30,  # 30天训练窗口
        step_size=15,    # 15天测试窗口
        n_trials=30,     # 每个窗口30次尝试
        metric='sharpe'
    )
    
    # 生成对比报告
    print("\n\n============= 优化方法对比报告 =============")
    
    # 提取各方法的关键指标
    methods = ["默认参数", "标准贝叶斯优化", "前推式贝叶斯优化"]
    results_list = [
        default_results, 
        bayes_results['final_results'], 
        wf_bayes_results['final_results']
    ]
    
    # 创建对比表格
    comparison_data = []
    for method, result in zip(methods, results_list):
        metrics = result['metrics']
        comparison_data.append({
            "方法": method,
            "总收益率(%)": metrics['total_return'],
            "年化收益率(%)": metrics['annual_return'] * 100,
            "最大回撤(%)": metrics['max_drawdown'],
            "夏普比率": metrics['sharpe'],
            "胜率(%)": metrics['win_rate'] * 100,
            "交易次数": metrics['trade_count'],
            "平均收益": metrics['avg_trade_return']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 绘制权益曲线对比
    plt.figure(figsize=(12, 6))
    for method, result in zip(methods, results_list):
        plt.plot(result['equity'], label=method)
    
    plt.title('不同优化方法的权益曲线对比', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('权益', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=300)
    plt.show()
    
    # 绘制回撤对比
    plt.figure(figsize=(12, 6))
    for method, result in zip(methods, results_list):
        equity = result['equity']
        drawdown = (equity.cummax() - equity) / equity.cummax() * 100
        plt.plot(drawdown, label=method)
    
    plt.title('不同优化方法的回撤对比', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('回撤(%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('drawdown_comparison.png', dpi=300)
    plt.show()
    
    # 保存对比结果到CSV
    comparison_df.to_csv('optimization_comparison.csv', index=False)
    
    print("\n对比报告已生成，图表已保存。")
    print("============================================")

if __name__ == "__main__":
    # 运行优化对比
    run_optimization_comparison()