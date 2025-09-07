import json
import re
import string
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Tuple
import tempfile
import os

def normalize_text(text: str) -> str:
    """标准化文本用于比较"""
    if not text:
        return None
        
    # Unicode标准化
    text = unicodedata.normalize("NFD", text)
    
    # 转小写
    text = text.lower()
    
    # 移除标点符号
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    
    # 移除冠词
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    
    # 清理空白字符
    text = " ".join(text.split())
    
    return text

def create_deep_copy(obj, save_method, load_method, llm):
    """创建对象的深度副本
    
    Args:
        obj: 要复制的对象
        save_method: 对象的保存方法
        load_method: 对象的加载方法
        llm: LLM实例，用于加载
        
    Returns:
        对象的深度副本
    """
    temp_path = tempfile.mktemp(suffix='.json')
    try:
        save_method(temp_path)
        return load_method(temp_path, llm)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def save_block_config(block, path: str, config: Dict[str, Any]):
    """保存block配置的通用方法"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def load_block_config(path: str) -> Dict[str, Any]:
    """加载block配置的通用方法"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_most_common_prediction(predictions: List[Dict[str, Any]], normalize_func=normalize_text) -> Tuple[str, Dict[str, Any]]:
    """获取最常见的预测结果
    
    Args:
        predictions: 预测结果列表
        normalize_func: 标准化函数
        
    Returns:
        (最常见的答案, 元数据)
    """
    # 标准化并统计
    normalized_predictions = [normalize_func(prediction['answer']) for prediction in predictions]
    normalized_predictions = [x for x in normalized_predictions if x is not None]

    # 如果没有有效预测
    if not normalized_predictions:
        if predictions:
            return predictions[0]['answer'], {
                "reasoning": predictions[0].get('reasoning', None), 
                "answer": predictions[0].get('answer', None),
                "all_predictions": predictions
            }
        else:
            return "", {
                "reasoning": "No valid predictions", 
                "answer": None,
                "all_predictions": []
            }

    # 找到最常见的标准化答案
    value_counts = Counter(normalized_predictions)
    most_common_normalized = value_counts.most_common(1)[0][0]

    # 返回对应的原始答案
    for prediction in predictions:
        if normalize_func(prediction['answer']) == most_common_normalized:
            return prediction['answer'], {
                "reasoning": prediction.get('reasoning', None), 
                "answer": prediction.get('answer', None),
                "all_predictions": predictions,
                "prediction_count": len(predictions),
                "most_common_count": value_counts.most_common(1)[0][1]
            }

    # 默认返回第一个预测
    return predictions[0]['answer'], {
        "reasoning": predictions[0].get('reasoning', None), 
        "answer": predictions[0].get('answer', None),
        "all_predictions": predictions,
        "prediction_count": len(predictions)
    }
