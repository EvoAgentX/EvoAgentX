#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票技术分析图表生成工具
为任意A股股票生成专业的技术分析图表和K线图
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')

class StockChartGenerator:
    """股票技术分析图表生成器"""
    
    def __init__(self, symbol: str, output_dir: str = "output"):
        """
        初始化图表生成器
        
        Args:
            symbol (str): 股票代码（如：300750、600519等）
            output_dir (str): 输出目录，默认为"output"
        """
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据缓存
        self.stock_data = None
        self.processed_data = None
    
    def generate_mock_data(self) -> pd.DataFrame:
        """生成模拟股票数据用于演示"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
        dates = [d for d in dates if d.weekday() < 5]  # 只保留工作日
        
        np.random.seed(42)
        base_price = 1500 if self.symbol == "600519" else 100
        
        prices = []
        current_price = base_price
        
        for i in range(len(dates)):
            change = np.random.normal(0, 0.02)
            current_price = current_price * (1 + change)
            prices.append(current_price)
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = close * 0.03
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.randint(100000, 1000000)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume,
            })
        
        df = pd.DataFrame(data)
        print(f"生成了 {len(df)} 条模拟数据")
        return df
    
    def get_stock_data(self) -> pd.DataFrame:
        """获取股票数据"""
        if self.stock_data is not None:
            return self.stock_data
        
        try:
            import akshare as ak
            print(f"获取股票 {self.symbol} 的数据...")
            
            try:
                df = ak.stock_zh_a_hist(symbol=self.symbol, period="daily", adjust="qfq")
            except:
                try:
                    formatted_symbol = f"sh{self.symbol}" if self.symbol.startswith('6') else f"sz{self.symbol}"
                    df = ak.stock_zh_a_hist(symbol=formatted_symbol, period="daily", adjust="qfq")
                except:
                    print("获取真实数据失败，使用模拟数据...")
                    return self.generate_mock_data()
            
            if df.empty:
                return self.generate_mock_data()
            
            # 重命名列
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close', 
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
            })
            
            print(f"成功获取 {len(df)} 条真实数据")
            self.stock_data = df.tail(250)  # 只保留最近250天的数据
            return self.stock_data
            
        except Exception as e:
            print(f"获取数据失败，使用模拟数据: {e}")
            return self.generate_mock_data()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 创建副本避免修改原数据
        df = df.copy()
        
        # 移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # 布林带
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # 填充NaN值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        self.processed_data = df
        return df
    
    def create_technical_chart(self) -> Optional[str]:
        """创建技术分析图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib import rcParams
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 获取处理后的数据
            if self.processed_data is None:
                df = self.get_stock_data()
                df = self.calculate_indicators(df)
            else:
                df = self.processed_data
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 创建图表
            fig, axes = plt.subplots(4, 1, figsize=(15, 20))
            fig.suptitle(f'{self.symbol} 技术分析图表', fontsize=16, fontweight='bold')
            
            # 1. 价格和移动平均线
            ax1 = axes[0]
            ax1.plot(df['date'], df['close'], label='收盘价', linewidth=2, color='blue')
            ax1.plot(df['date'], df['MA5'], label='MA5', alpha=0.8, color='orange')
            ax1.plot(df['date'], df['MA10'], label='MA10', alpha=0.8, color='green')
            ax1.plot(df['date'], df['MA20'], label='MA20', alpha=0.8, color='red')
            
            # 布林带
            ax1.fill_between(df['date'], df['BB_upper'], df['BB_lower'], alpha=0.1, color='gray', label='布林带')
            ax1.plot(df['date'], df['BB_upper'], alpha=0.5, color='gray', linestyle='--')
            ax1.plot(df['date'], df['BB_lower'], alpha=0.5, color='gray', linestyle='--')
            
            ax1.set_title('价格走势与技术指标')
            ax1.set_ylabel('价格 (元)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 成交量
            ax2 = axes[1]
            colors = ['red' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'green' 
                     for i in range(len(df))]
            ax2.bar(df['date'], df['volume'], color=colors, alpha=0.7)
            ax2.set_title('成交量')
            ax2.set_ylabel('成交量')
            ax2.grid(True, alpha=0.3)
            
            # 3. RSI
            ax3 = axes[2]
            ax3.plot(df['date'], df['RSI'], label='RSI', color='purple', linewidth=2)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线(70)')
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线(30)')
            ax3.fill_between(df['date'], 30, 70, alpha=0.1, color='yellow', label='正常区间')
            ax3.set_title('RSI指标')
            ax3.set_ylabel('RSI')
            ax3.set_ylim(0, 100)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. MACD
            ax4 = axes[3]
            ax4.plot(df['date'], df['MACD'], label='MACD', color='blue', linewidth=2)
            ax4.plot(df['date'], df['MACD_signal'], label='信号线', color='red', linewidth=2)
            
            # MACD柱状图
            colors = ['red' if x > 0 else 'green' for x in df['MACD_histogram']]
            ax4.bar(df['date'], df['MACD_histogram'], color=colors, alpha=0.6, label='MACD柱状图')
            
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_title('MACD指标')
            ax4.set_ylabel('MACD')
            ax4.set_xlabel('日期')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 格式化x轴日期
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = self.output_dir / f'{self.symbol}_technical_charts.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"📊 技术分析图表已保存: {chart_path}")
            return str(chart_path)
            
        except ImportError:
            print("⚠️ matplotlib未安装，跳过图表生成")
            return None
        except Exception as e:
            print(f"❌ 生成技术分析图表失败: {e}")
            return None
    
    def create_candlestick_chart(self) -> Optional[str]:
        """创建K线图（蜡烛图）"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Rectangle
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 获取处理后的数据
            if self.processed_data is None:
                df = self.get_stock_data()
                df = self.calculate_indicators(df)
            else:
                df = self.processed_data
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').tail(60)  # 只显示最近60天
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
            fig.suptitle(f'{self.symbol} K线图分析', fontsize=16, fontweight='bold')
            
            # 绘制K线图
            for i, row in df.iterrows():
                date = row['date']
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                # 确定颜色
                color = 'red' if close_price >= open_price else 'green'
                
                # 绘制高低价线
                ax1.plot([date, date], [low_price, high_price], color='black', linewidth=1)
                
                # 绘制实体
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 0.6, body_height,
                               facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.add_patch(rect)
            
            # 添加移动平均线
            ax1.plot(df['date'], df['MA5'], label='MA5', alpha=0.8, color='orange', linewidth=1.5)
            ax1.plot(df['date'], df['MA20'], label='MA20', alpha=0.8, color='blue', linewidth=1.5)
            
            ax1.set_title('K线图与移动平均线')
            ax1.set_ylabel('价格 (元)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 成交量图
            colors = ['red' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'green' 
                     for i in range(len(df))]
            ax2.bar(df['date'], df['volume'], color=colors, alpha=0.7, width=0.8)
            ax2.set_title('成交量')
            ax2.set_ylabel('成交量')
            ax2.set_xlabel('日期')
            ax2.grid(True, alpha=0.3)
            
            # 格式化x轴
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = self.output_dir / f'{self.symbol}_candlestick_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"📊 K线图已保存: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            print(f"❌ 生成K线图失败: {e}")
            return None
    
    def generate_all_charts(self) -> Dict[str, Optional[str]]:
        """生成所有类型的图表"""
        print(f"🚀 生成股票 {self.symbol} 的技术分析图表")
        print("=" * 60)
        
        print(f"📊 开始分析股票: {self.symbol}")
        
        # 1. 获取数据
        print("🔄 获取股票数据...")
        df = self.get_stock_data()
        if df is None:
            print("❌ 无法获取数据")
            return {}
        
        # 2. 计算技术指标
        print("🔢 计算技术指标...")
        self.calculate_indicators(df)
        
        # 3. 生成图表
        chart_paths = {}
        
        print("📊 生成技术分析图表...")
        technical_path = self.create_technical_chart()
        if technical_path:
            chart_paths['technical'] = technical_path
        
        print("🕯️ 生成K线图...")
        candlestick_path = self.create_candlestick_chart()
        if candlestick_path:
            chart_paths['candlestick'] = candlestick_path
        
        if chart_paths:
            print(f"✅ 图表生成成功:")
            for chart_type, path in chart_paths.items():
                print(f"   {chart_type}: {os.path.abspath(path)}")
        else:
            print("❌ 图表生成失败")
        
        return chart_paths


def generate_stock_charts(symbol: str = "300750", output_dir: str = "output", 
                         chart_types: List[str] = None) -> Dict[str, Optional[str]]:
    """
    生成股票技术分析图表的主函数
    
    Args:
        symbol (str): 股票代码（如：300750、000001、000858等）
        output_dir (str): 输出目录，默认为"output"
        chart_types (List[str]): 图表类型列表，可选 "technical", "candlestick"
                                默认生成所有类型
        
    Returns:
        Dict[str, Optional[str]]: 生成的图表路径字典
        
    Example:
        # 生成宁德时代的所有图表
        charts = generate_stock_charts("300750")
        
        # 只生成K线图
        charts = generate_stock_charts("600519", chart_types=["candlestick"])
        
        # 生成到指定目录
        charts = generate_stock_charts("000001", output_dir="my_charts")
    """
    if chart_types is None:
        chart_types = ["technical", "candlestick"]
    
    generator = StockChartGenerator(symbol, output_dir)
    
    # 如果需要生成所有类型，直接调用generate_all_charts
    if set(chart_types) == {"technical", "candlestick"}:
        return generator.generate_all_charts()
    
    # 否则按需生成
    print(f"🚀 生成股票 {symbol} 的指定图表类型")
    print("=" * 60)
    
    chart_paths = {}
    
    # 准备数据
    df = generator.get_stock_data()
    if df is None:
        print("❌ 无法获取数据")
        return {}
    
    generator.calculate_indicators(df)
    
    # 生成指定类型的图表
    if "technical" in chart_types:
        print("📊 生成技术分析图表...")
        technical_path = generator.create_technical_chart()
        if technical_path:
            chart_paths['technical'] = technical_path
    
    if "candlestick" in chart_types:
        print("🕯️ 生成K线图...")
        candlestick_path = generator.create_candlestick_chart()
        if candlestick_path:
            chart_paths['candlestick'] = candlestick_path
    
    if chart_paths:
        print(f"✅ 图表生成成功:")
        for chart_type, path in chart_paths.items():
            print(f"   {chart_type}: {os.path.abspath(path)}")
    else:
        print("❌ 图表生成失败")
    
    return chart_paths


def batch_generate_charts(symbols: List[str], output_base_dir: str = "charts") -> Dict[str, Dict]:
    """
    批量生成多个股票的图表
    
    Args:
        symbols (List[str]): 股票代码列表
        output_base_dir (str): 基础输出目录
        
    Returns:
        Dict[str, Dict]: 每个股票的生成结果
        
    Example:
        symbols = ["300750", "600519", "000001"]
        results = batch_generate_charts(symbols)
    """
    results = {}
    
    print(f"🚀 批量生成 {len(symbols)} 个股票的图表")
    print("=" * 60)
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n📈 [{i}/{len(symbols)}] 处理股票: {symbol}")
        print("-" * 40)
        
        try:
            # 为每个股票创建独立目录
            stock_output_dir = os.path.join(output_base_dir, f"stock_{symbol}")
            
            chart_paths = generate_stock_charts(
                symbol=symbol,
                output_dir=stock_output_dir,
                chart_types=["technical", "candlestick"]
            )
            
            results[symbol] = {
                'status': 'success',
                'charts': chart_paths,
                'output_dir': stock_output_dir
            }
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            results[symbol] = {
                'status': 'failed',
                'error': str(e),
                'charts': {},
                'output_dir': None
            }
    
    # 汇总结果
    print("\n" + "="*60)
    print("📋 批量生成结果汇总")
    print("="*60)
    
    success_count = 0
    for symbol, result in results.items():
        if result['status'] == 'success':
            success_count += 1
            print(f"✅ {symbol}: 成功生成 {len(result['charts'])} 个图表")
        else:
            print(f"❌ {symbol}: {result.get('error', '未知错误')}")
    
    print(f"\n🎉 批量生成完成: {success_count}/{len(symbols)} 成功")
    
    return results


if __name__ == "__main__":
    # 演示用法
    print("🚀 股票技术分析图表生成工具")
    print("=" * 60)
    
    # 示例1：生成单个股票的图表
    print("📊 示例1: 生成宁德时代图表")
    charts = generate_stock_charts("300750")
    
    # 示例2：只生成K线图
    print("\n📊 示例2: 只生成贵州茅台K线图")
    charts = generate_stock_charts("600519", chart_types=["candlestick"])
    
    # 示例3：批量生成
    print("\n📊 示例3: 批量生成多个股票图表")
    symbols = ["300750", "600519", "000001"]
    batch_results = batch_generate_charts(symbols, "batch_charts")
    
    print("\n💡 使用说明:")
    print("1. generate_stock_charts(symbol): 生成指定股票的所有图表")
    print("2. generate_stock_charts(symbol, chart_types=['candlestick']): 只生成K线图")
    print("3. batch_generate_charts(symbols): 批量生成多个股票图表") 