import json
import re
import string
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Tuple

class Aggregate:
    """聚合多个预测结果，返回最常见的答案"""
    
    def __init__(self, predictor, n: int = 3):
        """
        初始化聚合器
        
        Args:
            predictor: Predictor实例
            n: 生成预测的数量，默认3个
        """
        self.predictor = predictor
        self.n = n
        self.search_space = [1, 3, 5, 7, 9]  # 保留原有的搜索空间
    
    def __call__(self, problem, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """聚合多个预测结果，返回最常见的答案"""
        predictions = []

        # 生成 n 个预测
        for _ in range(self.n):
            prediction = self.predictor.execute(problem=problem, **kwargs)
            predictions.append(prediction)

        # 标准化并统计
        normalized_predictions = [self._normalize_text(prediction['answer']) for prediction in predictions]
        normalized_predictions = [x for x in normalized_predictions if x is not None]

        # 如果没有有效预测
        if not normalized_predictions:
            if predictions:
                return predictions[0]['answer'], {
                    "problem": problem, 
                    "reasoning": predictions[0].get('reasoning', None), 
                    "answer": predictions[0].get('answer', None),
                    "all_predictions": predictions
                }
            else:
                return "", {
                    "problem": problem, 
                    "reasoning": "No valid predictions", 
                    "answer": None,
                    "all_predictions": []
                }

        # 找到最常见的标准化答案
        value_counts = Counter(normalized_predictions)
        most_common_normalized = value_counts.most_common(1)[0][0]

        # 返回对应的原始答案
        for prediction in predictions:
            if self._normalize_text(prediction['answer']) == most_common_normalized:
                return prediction['answer'], {
                    "problem": problem, 
                    "reasoning": prediction.get('reasoning', None), 
                    "answer": prediction.get('answer', None),
                    "all_predictions": predictions,
                    "prediction_count": len(predictions),
                    "most_common_count": value_counts.most_common(1)[0][1]
                }

        # 默认返回第一个预测
        return predictions[0]['answer'], {
            "problem": problem, 
            "reasoning": predictions[0].get('reasoning', None), 
            "answer": predictions[0].get('answer', None),
            "all_predictions": predictions,
            "prediction_count": len(predictions)
        }
    
    def execute(self, problem, **kwargs) -> List[str]:
        """执行预测并返回所有结果"""
        predictions = []
        
        for _ in range(self.n):
            prediction = self.predictor.execute(problem=problem, **kwargs)
            predictions.append(prediction['answer'])

        return predictions
    
    def _normalize_text(self, text: str) -> str:
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
    
    def save(self, path: str):
        """保存聚合器配置"""
        params = {
            "predictor": self.predictor.prompt,
            "n": self.n,
            "search_space": self.search_space
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """加载聚合器配置"""
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
            self.predictor.prompt = params["predictor"]
            self.n = params.get("n", 3)
            self.search_space = params.get("search_space", [1, 3, 5, 7, 9])
    
    def get_registry(self) -> List[str]:
        """获取注册信息"""
        return ["aggregate.predictor.prompt", "aggregate.n", "aggregate.search_space"]

# 便捷创建函数
def create_aggregate(predictor, n: int = 3):
    """创建Aggregate的便捷函数
    
    Args:
        predictor: Predictor实例
        n: 生成预测的数量，默认3个
    """
    return Aggregate(predictor, n)

if __name__ == "__main__":
    # 测试代码
    import os
    from dotenv import load_dotenv
    from evoagentx.models import OpenAILLMConfig, OpenAILLM
    from predictor_agent import create_predictor_agent
    
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 创建LLM和predictor
    model_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=False, output_response=False)
    llm = OpenAILLM(config=model_config)
    predictor = create_predictor_agent(llm)
    
    # 创建aggregate
    aggregate = create_aggregate(predictor, n=3)
    
    # 测试执行
    problem = "What is 15 + 27?"
    result, metadata = aggregate(problem)
    print(f"Problem: {problem}")
    print(f"Aggregated Result: {result}")
    print(f"Prediction Count: {metadata['prediction_count']}")
    print(f"Most Common Count: {metadata.get('most_common_count', 'N/A')}")
    print(f"All Predictions: {[p['answer'] for p in metadata['all_predictions']]}")
    
    # 测试保存和加载
    save_path = "examples/mass/blocks/saved_aggregate.json"
    aggregate.save(save_path)
    print(f"\nAggregate saved to {save_path}")
    
    # 测试加载
    new_aggregate = create_aggregate(predictor)
    new_aggregate.load(save_path)
    print(f"Loaded aggregate with n={new_aggregate.n}")
