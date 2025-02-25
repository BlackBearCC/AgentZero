import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import json
import os
from datetime import datetime

from src.trading.ai_run_backtest import TransformerBacktester

class HyperparameterOptimizer:
    """Transformer模型超参数优化器"""
    def __init__(self, model_path: str, data: pd.DataFrame):
        self.model_path = model_path
        self.data = data
        self.backtester = TransformerBacktester(model_path)
        
    def grid_search(self, 
                   param_grid: Dict[str, List], 
                   metric: str = 'sharpe',
                   n_jobs: int = -1,
                   verbose: bool = True) -> pd.DataFrame:
        """
        网格搜索超参数优化
        
        参数:
            param_grid: 参数网格，如 {'position_limit': [0.3, 0.5, 0.7], 'signal_threshold': [0.2, 0.3, 0.4]}
            metric: 优化指标，可选 'sharpe', 'return', 'drawdown', 'calmar'
            n_jobs: 并行任务数，-1表示使用所有CPU
            verbose: 是否显示详细进度
            
        返回:
            包含所有参数组合结果的DataFrame
        """
        # 生成所有参数组合
        keys = param_grid.keys()
        param_combinations = list(itertools.product(*param_grid.values()))
        param_dicts = [dict(zip(keys, combo)) for combo in param_combinations]
        
        print(f"开始网格搜索，共{len(param_dicts)}种参数组合...")
        
        # 创建进度记录文件
        progress_file = "optimization_progress.csv"
        with open(progress_file, 'w') as f:
            header = list(keys) + ['sharpe', 'annual_return', 'max_drawdown', 'win_rate', 'trade_count', 'timestamp']
            f.write(','.join(header) + '\n')
        
        # 并行执行回测
        if n_jobs == 1:
            # 单线程执行，便于查看进度
            results = []
            for i, params in enumerate(param_dicts):
                print(f"\n测试参数组合 {i+1}/{len(param_dicts)}: {params}")
                result = self._evaluate_params(params, verbose=verbose)
                results.append(result)
                
                # 保存进度到文件
                with open(progress_file, 'a') as f:
                    values = [str(result.get(k, '')) for k in header]
                    f.write(','.join(values) + '\n')
                
                # 打印当前最佳结果
                temp_results = pd.DataFrame(results)
                if not temp_results.empty:
                    best_idx = temp_results[metric].idxmax()
                    print(f"\n当前最佳参数 (基于{metric}):")
                    for k, v in temp_results.iloc[best_idx].items():
                        print(f"  {k}: {v}")
        else:
            # 并行执行
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._evaluate_params)(params, verbose=False) for params in tqdm(param_dicts)
            )
        
        # 整理结果
        results_df = pd.DataFrame(results)
        
        # 根据指定指标排序
        if metric == 'sharpe':
            results_df = results_df.sort_values('sharpe', ascending=False)
        elif metric == 'return':
            results_df = results_df.sort_values('annual_return', ascending=False)
        elif metric == 'drawdown':
            results_df = results_df.sort_values('max_drawdown', ascending=True)
        elif metric == 'calmar':
            results_df['calmar'] = results_df['annual_return'] / (results_df['max_drawdown']/100)
            results_df = results_df.sort_values('calmar', ascending=False)
        
        print("超参数优化完成！")
        print(f"最优参数组合:\n{results_df.iloc[0]}")
        
        # 保存结果到CSV和JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f'optimization_results_{timestamp}.csv', index=False)
        
        # 保存最优参数
        best_params = results_df.iloc[0].to_dict()
        params_to_save = {k: v for k, v in best_params.items() 
                         if k not in ['sharpe', 'annual_return', 'max_drawdown', 
                                     'win_rate', 'trade_count', 'timestamp']}
        
        with open(f'best_params_{timestamp}.json', 'w') as f:
            json.dump(params_to_save, f, indent=4)
        
        print(f"结果已保存到 optimization_results_{timestamp}.csv")
        print(f"最优参数已保存到 best_params_{timestamp}.json")
        
        return results_df
    
    def _evaluate_params(self, params: Dict, verbose: bool = False) -> Dict:
        """评估单个参数组合"""
        try:
            # 运行回测
            results = self.backtester.run_backtest(
                data=self.data,
                verbose=verbose,  # 控制详细输出
                **params
            )
            
            # 提取指标
            metrics = results['metrics']
            
            # 正确计算夏普比率 (年化收益率/年化波动率)
            # 确保使用年化值计算夏普比率
            annual_return = metrics['annual_return']
            annual_vol = metrics.get('annual_vol', 0.01)  # 防止除零错误
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            
            # 合并参数和指标
            result = {
                **params,
                'sharpe': sharpe,
                'annual_return': annual_return * 100,
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'] * 100,
                'trade_count': metrics['trade_count'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if verbose:
                print(f"参数评估结果: 夏普比率={sharpe:.2f}, 年化收益={annual_return*100:.2f}%, 最大回撤={metrics['max_drawdown']:.2f}%")
            
            return result
        except Exception as e:
            print(f"参数评估失败: {params}, 错误: {e}")
            return {
                **params,
                'sharpe': -999,
                'annual_return': -999,
                'max_drawdown': 100,
                'win_rate': 0,
                'trade_count': 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def plot_param_impact(self, results_df: pd.DataFrame, param_name: str, metric: str = 'sharpe'):
        """绘制单个参数对性能指标的影响"""
        plt.figure(figsize=(10, 6))
        
        # 按参数分组并计算平均指标
        grouped = results_df.groupby(param_name)[metric].mean()
        
        plt.plot(grouped.index, grouped.values, 'o-', linewidth=2)
        plt.title(f'{param_name}对{metric}的影响')
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.grid(True)
        plt.savefig(f'param_impact_{param_name}_{metric}.png')
        plt.show()
    
    def plot_heatmap(self, results_df: pd.DataFrame, param1: str, param2: str, metric: str = 'sharpe'):
        """绘制两个参数的热力图"""
        pivot = results_df.pivot_table(
            index=param1, 
            columns=param2, 
            values=metric,
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot, cmap='viridis', aspect='auto')
        plt.colorbar(label=metric)
        plt.title(f'{param1}和{param2}对{metric}的影响')
        plt.xlabel(param2)
        plt.ylabel(param1)
        
        # 添加数值标签
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                plt.text(j, i, f'{pivot.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white')
        
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.savefig(f'heatmap_{param1}_{param2}_{metric}.png')
        plt.show()
    
    def walk_forward_optimization(self, 
                                param_grid: Dict[str, List],
                                window_size: int = 60,  # 天数
                                step_size: int = 30,    # 天数
                                metric: str = 'sharpe',
                                verbose: bool = True) -> pd.DataFrame:
        """
        前推式优化 - 在滚动时间窗口上优化参数
        
        参数:
            param_grid: 参数网格
            window_size: 每个窗口的天数
            step_size: 窗口滚动步长(天数)
            metric: 优化指标
            verbose: 是否显示详细进度
            
        返回:
            每个时间窗口的最优参数
        """
        # 将数据按日期分割成多个窗口
        dates = self.data.index
        start_date = dates[0]
        end_date = dates[-1]
        
        # 生成时间窗口
        windows = []
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + pd.Timedelta(days=window_size)
            if current_end > end_date:
                current_end = end_date
            
            windows.append((current_start, current_end))
            current_start += pd.Timedelta(days=step_size)
        
        print(f"前推式优化: 共{len(windows)}个时间窗口")
        
        # 创建进度记录文件
        wfo_progress_file = "wfo_progress.csv"
        with open(wfo_progress_file, 'w') as f:
            header = ['window_start', 'window_end'] + list(param_grid.keys()) + ['sharpe', 'annual_return', 'max_drawdown']
            f.write(','.join(header) + '\n')
        
        # 在每个窗口上进行优化
        window_results = []
        for i, (start, end) in enumerate(windows):
            print(f"\n优化窗口 {i+1}/{len(windows)}: {start.date()} 到 {end.date()}")
            
            # 提取窗口数据
            window_data = self.data[(self.data.index >= start) & (self.data.index <= end)]
            
            # 创建新的回测器
            window_backtester = TransformerBacktester(self.model_path)
            
            # 在当前窗口上进行网格搜索
            keys = param_grid.keys()
            param_combinations = list(itertools.product(*param_grid.values()))
            param_dicts = [dict(zip(keys, combo)) for combo in param_combinations]
            
            # 评估所有参数组合
            window_param_results = []
            for j, params in enumerate(param_dicts):
                if verbose:
                    print(f"  测试参数组合 {j+1}/{len(param_dicts)}: {params}")
                
                try:
                    results = window_backtester.run_backtest(
                        data=window_data,
                        verbose=False,
                        **params
                    )
                    
                    metrics = results['metrics']
                    # 正确计算夏普比率
                    annual_return = metrics['annual_return']
                    annual_vol = metrics.get('annual_vol', 0.01)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    result = {
                        **params,
                        'sharpe': sharpe,
                        'annual_return': annual_return * 100,
                        'max_drawdown': metrics['max_drawdown'],
                        'win_rate': metrics['win_rate'] * 100
                    }
                    window_param_results.append(result)
                    
                    if verbose:
                        print(f"    结果: 夏普比率={sharpe:.2f}, 年化收益={annual_return*100:.2f}%, 最大回撤={metrics['max_drawdown']:.2f}%")
                        
                except Exception as e:
                    print(f"    窗口参数评估失败: {params}, 错误: {e}")
            
            # 找出最优参数
            if window_param_results:
                results_df = pd.DataFrame(window_param_results)
                best_params = results_df.sort_values(metric, ascending=False).iloc[0].to_dict()
                best_params['window_start'] = start
                best_params['window_end'] = end
                window_results.append(best_params)
                
                # 保存进度到文件
                with open(wfo_progress_file, 'a') as f:
                    values = [str(start), str(end)]
                    for k in param_grid.keys():
                        values.append(str(best_params.get(k, '')))
                    values.extend([
                        str(best_params.get('sharpe', '')),
                        str(best_params.get('annual_return', '')),
                        str(best_params.get('max_drawdown', ''))
                    ])
                    f.write(','.join(values) + '\n')
                
                print(f"\n窗口 {i+1} 最优参数:")
                for k, v in best_params.items():
                    if k not in ['window_start', 'window_end']:
                        print(f"  {k}: {v}")
        
        # 返回每个窗口的最优参数
        wfo_df = pd.DataFrame(window_results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wfo_df.to_csv(f'wfo_results_{timestamp}.csv', index=False)
        
        # 计算并保存平均最优参数
        avg_params = {}
        for param in param_grid.keys():
            if param in wfo_df.columns:
                avg_params[param] = wfo_df[param].mean()
        
        with open(f'wfo_avg_params_{timestamp}.json', 'w') as f:
            json.dump(avg_params, f, indent=4)
        
        print(f"\n前推式优化完成！结果已保存到 wfo_results_{timestamp}.csv")
        print(f"平均最优参数已保存到 wfo_avg_params_{timestamp}.json")
        
        return wfo_df

# 使用示例
if __name__ == "__main__":
    from src.trading.feeds.crypto_feed import DataManager
    
    # 加载数据
    data_mgr = DataManager()
    data = data_mgr.get_feed(
        symbol='BTC/USDT',
        timeframe='15m',
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31)
    )['indicator']
    
    # 创建优化器
    optimizer = HyperparameterOptimizer('models/alpha_v1.pth', data)
    
    # 定义参数网格
    param_grid = {
        'position_limit': [0.3, 0.5, 0.7],
        'signal_threshold': [0.2, 0.3, 0.4],
        'volatility_threshold': [0.01, 0.015, 0.02],
        'smoothing_factor': [0.5, 0.7, 0.9],
        'stop_loss_pct': [0.01, 0.02, 0.03],
        'take_profit_pct': [0.03, 0.05, 0.07]
    }
    
    # 运行网格搜索
    results = optimizer.grid_search(param_grid, metric='sharpe', n_jobs=4)
    
    # 可视化参数影响
    optimizer.plot_param_impact(results, 'signal_threshold')
    optimizer.plot_heatmap(results, 'position_limit', 'signal_threshold')
    
    # 运行前推式优化
    wfo_results = optimizer.walk_forward_optimization(
        param_grid={
            'position_limit': [0.3, 0.5, 0.7],
            'signal_threshold': [0.2, 0.3, 0.4]
        },
        window_size=30,
        step_size=15
    )
    
    print("前推式优化结果:")
    print(wfo_results) 