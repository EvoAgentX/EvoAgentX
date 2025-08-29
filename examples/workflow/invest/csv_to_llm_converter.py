#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV数据转LLM JSON格式转换器
将股票数据CSV文件转换为适合LLM分析的JSON格式
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

class CSVToLLMConverter:
    """CSV转LLM JSON格式转换器"""
    
    def __init__(self, data_dir: str):
        """
        初始化转换器
        
        Args:
            data_dir (str): 数据目录路径（如 output_300750）
        """
        self.data_dir = Path(data_dir)
        
        # 文件优先级和行数配置
        self.file_priority = {
            'stock_daily_catl': {'weight': 'high', 'max_rows': 30},
            'institution_recommendation_catl': {'weight': 'high', 'max_rows': 20},
            'stock_news_catl': {'weight': 'high', 'max_rows': 15},
            'china_cpi': {'weight': 'medium', 'max_rows': 10},
            'china_gdp': {'weight': 'medium', 'max_rows': 10},
            'industry_fund_flow': {'weight': 'medium', 'max_rows': 15},
            'market_overview': {'weight': 'normal', 'max_rows': 5},
            'regional_indices': {'weight': 'normal', 'max_rows': 10},
            'option_volatility': {'weight': 'normal', 'max_rows': 8},
            'fund_flow_industry': {'weight': 'normal', 'max_rows': 12}
        }
    
    def find_csv_files(self) -> Dict[str, Dict]:
        """查找并分类CSV文件"""
        csv_files = {}
        
        if not self.data_dir.exists():
            print(f"❌ 数据目录不存在: {self.data_dir}")
            return csv_files
        
        for file_path in self.data_dir.glob("*.csv"):
            filename = file_path.name
            
            # 跳过collection_report文件
            if 'collection_report' in filename.lower():
                continue
            
            # 通过文件名识别数据类型
            file_type = self._identify_file_type(filename)
            if file_type:
                csv_files[file_type] = {
                    'file_path': file_path,
                    'filename': filename,
                    'config': self.file_priority.get(file_type, {'weight': 'normal', 'max_rows': 10})
                }
        
        return csv_files
    
    def _identify_file_type(self, filename: str) -> Optional[str]:
        """根据文件名识别数据类型"""
        filename_lower = filename.lower()
        
        # 定义文件名关键词映射
        type_mapping = {
            'stock_daily_catl': ['stock_daily'],
            'institution_recommendation_catl': ['institution_recommendation'],
            'stock_news_catl': ['stock_news'],
            'china_cpi': ['china_cpi'],
            'china_gdp': ['china_gdp'],
            'industry_fund_flow': ['industry_fund_flow'],
            'market_overview': ['market_overview'],
            'regional_indices': ['regional_indices'],
            'option_volatility': ['option_volatility'],
            'fund_flow_industry': ['fund_flow_industry']
        }
        
        for file_type, keywords in type_mapping.items():
            if any(keyword in filename_lower for keyword in keywords):
                return file_type
        
        return None
    
    def read_and_process_csv(self, file_path: Path, max_rows: int, weight: str) -> List[Dict]:
        """读取并处理CSV文件"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            if df.empty:
                print(f"⚠️ 文件为空: {file_path.name}")
                return []
            
            # 根据权重选择数据行
            if weight == 'high':
                # 高优先级：取最新的数据（末尾）
                processed_df = df.tail(max_rows)
            else:
                # 其他优先级：取开头的数据
                processed_df = df.head(max_rows)
            
            # 填充NaN值
            processed_df = processed_df.fillna('')
            
            # 转换为字典列表
            records = processed_df.to_dict(orient='records')
            
            print(f"✅ 处理完成 {file_path.name}: {len(records)} 条记录")
            return records
            
        except Exception as e:
            print(f"❌ 处理文件失败 {file_path.name}: {e}")
            return []
    
    def generate_llm_analysis_prompt(self) -> str:
        """生成适合LLM分析的提示格式"""
        csv_files = self.find_csv_files()
        
        if not csv_files:
            return "No valid CSV files found in the specified directory."
        
        # 按权重排序，股票日线数据优先
        def sort_priority(item):
            file_type, file_info = item
            weight = file_info['config']['weight']
            
            # 股票日线数据最优先
            if 'stock_daily_catl' in file_type:
                return (0, 0)  # 最高优先级
            
            weight_order = {'high': 1, 'medium': 2, 'normal': 3}
            base_priority = weight_order.get(weight, 4)
            
            # 在同权重内，按文件类型细分
            if weight == 'high':
                if 'institution_recommendation' in file_type:
                    return (base_priority, 1)
                elif 'stock_news' in file_type:
                    return (base_priority, 2)
            
            return (base_priority, 0)
        
        sorted_files = sorted(csv_files.items(), key=sort_priority)
        
        # 构建LLM分析提示
        prompt_parts = []
        
        # 添加总体说明
        stock_code = self._extract_stock_code()
        prompt_parts.append(f"# 股票 {stock_code} 综合数据分析")
        prompt_parts.append("\n以下是该股票的各类数据，请进行综合分析并给出投资建议：\n")
        
        # 添加数据概览
        prompt_parts.append("## 📊 数据概览")
        for i, (file_type, file_info) in enumerate(sorted_files, 1):
            weight_emoji = {"high": "🔥", "medium": "⭐", "normal": "📋"}
            emoji = weight_emoji.get(file_info['config']['weight'], "📋")
            prompt_parts.append(f"{i}. {emoji} {self._get_chinese_name(file_type)} ({file_info['filename']})")
        
        prompt_parts.append("\n## 📈 详细数据\n")
        
        # 添加每个数据集
        for i, (file_type, file_info) in enumerate(sorted_files, 1):
            file_path = file_info['file_path']
            config = file_info['config']
            
            # 读取和处理数据
            data = self.read_and_process_csv(file_path, config['max_rows'], config['weight'])
            
            if not data:
                continue
            
            # 添加数据集标题
            chinese_name = self._get_chinese_name(file_type)
            priority_label = {"high": "(重点关注)", "medium": "(重要参考)", "normal": "(背景信息)"}
            priority = priority_label.get(config['weight'], "")
            
            prompt_parts.append(f"### Dataset {i}: {chinese_name} {priority}")
            prompt_parts.append(f"文件: {file_info['filename']}")
            prompt_parts.append(f"数据量: {len(data)} 条记录\n")
            
            # 添加JSON数据
            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            prompt_parts.append("```json")
            prompt_parts.append(json_data)
            prompt_parts.append("```\n")
        
        # 添加分析要求
        prompt_parts.append("## 🎯 分析要求")
        prompt_parts.append("请基于以上数据进行以下分析：")
        prompt_parts.append("1. **价格趋势分析**: 根据股票日线数据分析价格走势")
        prompt_parts.append("2. **技术指标评估**: 结合移动平均线、成交量等技术指标")
        prompt_parts.append("3. **机构观点**: 分析机构评级和目标价")
        prompt_parts.append("4. **市场环境**: 考虑宏观经济数据和行业资金流向")
        prompt_parts.append("5. **新闻影响**: 评估相关新闻对股价的潜在影响")
        prompt_parts.append("6. **投资建议**: 给出明确的买入/持有/卖出建议及理由")
        prompt_parts.append("\n请用中文回答，并提供具体的数据支撑。")
        
        return "\n".join(prompt_parts)
    
    def _extract_stock_code(self) -> str:
        """从目录名提取股票代码"""
        dir_name = self.data_dir.name
        if 'output_' in dir_name:
            return dir_name.replace('output_', '')
        return dir_name
    
    def _get_chinese_name(self, file_type: str) -> str:
        """获取数据类型的中文名称"""
        name_mapping = {
            'stock_daily_catl': 'Stock Daily Price Data (股票日线数据)',
            'institution_recommendation_catl': 'Institution Recommendations (机构评级)',
            'stock_news_catl': 'Stock News (股票新闻)',
            'china_cpi': 'China CPI (中国CPI)',
            'china_gdp': 'China GDP (中国GDP)',
            'industry_fund_flow': 'Industry Fund Flow (行业资金流)',
            'market_overview': 'Market Overview (市场概况)',
            'regional_indices': 'Regional Indices (区域指数)',
            'option_volatility': 'Option Volatility (期权波动率)',
            'fund_flow_industry': 'Fund Flow Industry (行业资金流向)'
        }
        return name_mapping.get(file_type, file_type)
    
    def save_prompt_to_file(self, output_path: str = None) -> str:
        """保存提示内容到文件"""
        if output_path is None:
            output_path = self.data_dir / "llm_analysis_prompt.txt"
        
        prompt_content = self.generate_llm_analysis_prompt()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt_content)
            
            file_size = os.path.getsize(output_path)
            print(f"✅ LLM分析提示已保存: {output_path}")
            print(f"📄 文件大小: {file_size:,} 字节")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
            return ""
    
    def get_json_data(self) -> Dict[str, List[Dict]]:
        """直接获取JSON格式的数据字典"""
        csv_files = self.find_csv_files()
        json_data = {}
        
        for file_type, file_info in csv_files.items():
            config = file_info['config']
            data = self.read_and_process_csv(
                file_info['file_path'], 
                config['max_rows'], 
                config['weight']
            )
            if data:
                chinese_name = self._get_chinese_name(file_type)
                json_data[chinese_name] = data
        
        return json_data


def convert_csv_to_llm_json(data_dir: str, output_file: str = None) -> str:
    """
    快速转换CSV数据为LLM JSON格式的主函数
    
    Args:
        data_dir (str): 数据目录路径（如 output_300750）
        output_file (str): 输出文件路径（可选）
        
    Returns:
        str: 生成的提示文件路径
        
    Example:
        convert_csv_to_llm_json("output_300750")
        convert_csv_to_llm_json("output_600519", "my_prompt.txt")
    """
    print(f"🔄 开始转换 {data_dir} 中的CSV数据...")
    
    converter = CSVToLLMConverter(data_dir)
    result_path = converter.save_prompt_to_file(output_file)
    
    if result_path:
        print(f"✅ 转换完成: {os.path.abspath(result_path)}")
    else:
        print("❌ 转换失败")
    
    return result_path


def get_stock_data_json(data_dir: str) -> Dict[str, List[Dict]]:
    """
    获取股票数据的JSON格式字典
    
    Args:
        data_dir (str): 数据目录路径（如 output_300750）
        
    Returns:
        Dict[str, List[Dict]]: 包含所有数据的字典
        
    Example:
        data = get_stock_data_json("output_300750")
        print(data.keys())  # 查看所有数据类型
    """
    converter = CSVToLLMConverter(data_dir)
    return converter.get_json_data()


if __name__ == "__main__":
    # 演示用法
    print("🚀 CSV转LLM JSON格式转换器")
    print("=" * 50)
    
    # 示例：转换宁德时代数据
    example_dir = "output_300750"
    
    if os.path.exists(example_dir):
        print(f"📊 转换示例: {example_dir}")
        
        # 方法1：生成LLM提示文件
        prompt_file = convert_csv_to_llm_json(example_dir)
        
        # 方法2：直接获取JSON数据
        json_data = get_stock_data_json(example_dir)
        print(f"\n📋 获取到 {len(json_data)} 个数据类型:")
        for data_type in json_data.keys():
            print(f"   📊 {data_type}")
    
    else:
        print(f"⚠️ 示例目录不存在: {example_dir}")
        print("请先运行数据收集程序生成股票数据")
    
    print("\n💡 使用方法:")
    print("1. convert_csv_to_llm_json('output_股票代码')")
    print("2. get_stock_data_json('output_股票代码')") 