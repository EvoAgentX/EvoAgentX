import json
import re
import string
import unicodedata
from collections import Counter
from evoagentx.workflow.blocks.block import block

class aggregate(block):
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.search_space = [1,3,5,7,9]
    
    def __call__(self, problem, **kwargs):
        """聚合多个预测结果，返回最常见的答案"""
        predictions = []

        # 生成 n 个预测
        for _ in range(3):
            prediction = self.predictor.execute(problem=problem)
            predictions.append(prediction)

        # 标准化并统计
        normalized_predictions = [self._normalize_text(prediction['answer']) for prediction in predictions]
        normalized_predictions = [x for x in normalized_predictions if x is not None]

        # 如果没有有效预测
        if not normalized_predictions:
            if predictions:
                return predictions[0]['answer'], {"problem": problem, "reasoning": predictions[0].get('reasoning', None), "answer": predictions[0].get('answer', None)}
            else:
                return "", {"problem": problem, "reasoning": "No valid predictions", "answer": None}

        # 找到最常见的标准化答案
        value_counts = Counter(normalized_predictions)
        most_common_normalized = value_counts.most_common(1)[0][0]

        # 返回对应的原始答案
        for prediction in predictions:
            if self._normalize_text(prediction['answer']) == most_common_normalized:
                return prediction['answer'], {"problem": problem, "reasoning": prediction.get('reasoning', None), "answer": prediction.get('answer', None)}

        # 默认返回第一个预测
        return predictions[0]['answer'], {"problem": problem, "reasoning": predictions[0].get('reasoning', None), "answer": predictions[0].get('answer', None)}
    
    def execute(self, problem, **kwargs):
        """执行预测并返回所有结果"""
        predictions = []
        
        for _ in range(self.n):
            prediction = self.predictor.execute(problem = problem, **kwargs)
            predictions.append(prediction['answer'])

        return predictions
    
    def _normalize_text(self, text):
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
        params = {"predictor": self.predictor.prompt}
        
        with open(path, "w") as f:
            json.dump(params, f)
    
    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.predictor.prompt = params["predictor"]
        
    def get_registry(self):
        return ["aggregater.predictor.prompt"]