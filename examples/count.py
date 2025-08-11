import os
import json

# 动态定位脚本目录，保证相对路径正确
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "number.json")

try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print("题目数量：", len(data.get("examples", [])))
except Exception:
    # 如果 JSON 解析失败，按行统计 input 字段出现次数
    count = 0
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if '"input"' in line:
                count += 1
    print("题目数量（按行统计）：", count)