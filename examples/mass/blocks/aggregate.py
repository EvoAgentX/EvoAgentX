import json
from typing import List, Dict, Any, Tuple
from utils import normalize_text, get_most_common_prediction, create_deep_copy

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

        # 使用utils中的函数获取最常见预测
        answer, metadata = get_most_common_prediction(predictions, normalize_text)
        metadata["problem"] = problem
        return answer, metadata
    
    def execute(self, problem, **kwargs) -> List[str]:
        """执行预测并返回所有结果"""
        predictions = []
        
        for _ in range(self.n):
            prediction = self.predictor.execute(problem=problem, **kwargs)
            predictions.append(prediction['answer'])

        return predictions
    

    
    def save(self, path: str):
        """保存聚合器配置"""
        params = {
            "predictor": self.predictor.agent.prompt,
            "n": self.n,
            "search_space": self.search_space
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """加载聚合器配置"""
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
            self.predictor.agent.prompt = params["predictor"]
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


