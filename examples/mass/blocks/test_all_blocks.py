import os
import tempfile
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from predictor_agent import create_predictor_agent
from aggregate import create_aggregate
from debate import create_debate_agent
from reflect import create_reflect_agent
from summarize import create_summarize_agent
from execute import create_execute_agent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def test_all_blocks():
    """测试所有blocks的功能"""
    
    # 创建LLM
    model_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=False, output_response=False)
    llm = OpenAILLM(config=model_config)
    
    print("=== 测试 Predictor ===")
    predictor = create_predictor_agent(llm)
    problem = "What is 15 + 27?"
    result, metadata = predictor(problem)
    print(f"Problem: {problem}")
    print(f"Result: {result}")
    print(f"Reasoning: {metadata['reasoning'][:100]}...")
    
    print("\n=== 测试 Aggregate ===")
    aggregate = create_aggregate(predictor, n=3)
    result, metadata = aggregate(problem)
    print(f"Problem: {problem}")
    print(f"Aggregated Result: {result}")
    print(f"Prediction Count: {metadata['prediction_count']}")
    
    print("\n=== 测试 Debate ===")
    debate = create_debate_agent(llm)
    solutions = ["42", "41", "43"]
    result, metadata = debate(problem, solutions)
    print(f"Problem: {problem}")
    print(f"Solutions: {solutions}")
    print(f"Selected: {result}")
    print(f"Index: {metadata['index']}")
    
    print("\n=== 测试 Reflect ===")
    reflect = create_reflect_agent(llm)
    text = "The answer is 42"
    result, metadata = reflect(problem, text)
    print(f"Problem: {problem}")
    print(f"Text: {text}")
    print(f"Correct: {result}")
    print(f"Feedback: {metadata['feedback'][:100]}...")
    
    print("\n=== 测试 Summarize ===")
    summarize = create_summarize_agent(llm)
    context = "Mathematics is the study of numbers and operations. Addition is a basic operation. 15 + 27 equals 42."
    result, metadata = summarize(problem, context)
    print(f"Problem: {problem}")
    print(f"Context: {context}")
    print(f"Summary: {result}")
    
    print("\n=== 测试 Execute ===")
    execute = create_execute_agent(llm)
    question = "Write a function that adds two numbers"
    solution = "def add(a, b):\n    return a + b\n\nprint(add(15, 27))"
    result, metadata = execute(question, solution)
    print(f"Question: {question}")
    print(f"Solution: {solution}")
    print(f"Correct: {result}")
    print(f"Traceback: {metadata['traceback'][:100]}...")
    
    print("\n=== 测试保存和加载 ===")
    # 使用临时文件进行保存和加载测试
    temp_files = {}
    
    try:
        # 创建临时文件
        temp_files = {
            "predictor": tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False),
            "aggregate": tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False),
            "debate": tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False),
            "reflect": tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False),
            "summarize": tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False),
            "execute": tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        }
        
        # 保存所有agents
        predictor.save(temp_files["predictor"].name)
        aggregate.save(temp_files["aggregate"].name)
        debate.save(temp_files["debate"].name)
        reflect.save(temp_files["reflect"].name)
        summarize.save(temp_files["summarize"].name)
        execute.save(temp_files["execute"].name)
        
        print("All agents saved successfully!")
        
        # 测试加载
        loaded_predictor = predictor.load(temp_files["predictor"].name, llm)
        loaded_debate = debate.load(temp_files["debate"].name, llm)
        loaded_reflect = reflect.load(temp_files["reflect"].name, llm)
        loaded_summarize = summarize.load(temp_files["summarize"].name, llm)
        loaded_execute = execute.load(temp_files["execute"].name, llm)
        
        print("All agents loaded successfully!")
        
    finally:
        # 清理临时文件
        for temp_file in temp_files.values():
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        print("Temporary files cleaned up!")

if __name__ == "__main__":
    test_all_blocks()
