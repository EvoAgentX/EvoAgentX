import os
import tempfile
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from mass_optimizer import MassOptimiser, create_mass_workflow, MassWorkflow

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def test_workflow_creation():
    """测试工作流创建"""
    print("=== 测试工作流创建 ===")
    
    # 创建LLM
    config = OpenAILLMConfig(
        model="gpt-4o-mini", 
        openai_key=OPENAI_API_KEY, 
        stream=False, 
        output_response=False
    )
    llm = OpenAILLM(config=config)
    
    # 创建工作流
    workflow = create_mass_workflow(llm)
    
    print(f"工作流创建成功，包含 {len(workflow.blocks)} 个blocks")
    for i, block in enumerate(workflow.blocks):
        print(f"  Block {i}: {block.name} (n={block.n}, activate={block.activate})")
    
    return workflow


def test_workflow_execution(workflow):
    """测试工作流执行"""
    print("\n=== 测试工作流执行 ===")
    
    # 激活一些blocks进行测试
    workflow.blocks[0].activate = True  # predictor
    workflow.blocks[0].n = 1
    workflow.blocks[1].activate = True  # aggregate
    workflow.blocks[1].n = 1
    
    test_problem = "What is 15 + 27?"
    print(f"测试问题: {test_problem}")
    
    try:
        result, metadata = workflow(test_problem)
        print(f"执行结果: {result}")
        print(f"元数据: {metadata}")
        return True
    except Exception as e:
        print(f"执行失败: {e}")
        return False


def test_workflow_save_load(workflow):
    """测试工作流保存和加载"""
    print("\n=== 测试工作流保存和加载 ===")
    
    # 使用临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # 保存工作流
        workflow.save(temp_path)
        print(f"工作流已保存到: {temp_path}")
        
        # 创建新工作流并加载
        config = OpenAILLMConfig(
            model="gpt-4o-mini", 
            openai_key=OPENAI_API_KEY, 
            stream=False, 
            output_response=False
        )
        llm = OpenAILLM(config=config)
        new_workflow = create_mass_workflow(llm)
        
        # 加载配置
        new_workflow.load(temp_path)
        print("工作流加载成功")
        
        # 验证加载的配置
        for i, block in enumerate(new_workflow.blocks):
            print(f"  Block {i}: {block.name} (n={block.n}, activate={block.activate})")
        
        return True
        
    except Exception as e:
        print(f"保存/加载失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print("临时文件已清理")


def test_mass_optimizer_creation():
    """测试MASS优化器创建"""
    print("\n=== 测试MASS优化器创建 ===")
    
    # 创建LLM
    executor_config = OpenAILLMConfig(
        model="gpt-4o-mini", 
        openai_key=OPENAI_API_KEY, 
        stream=False, 
        output_response=False
    )
    executor_llm = OpenAILLM(config=executor_config)
    
    optimizer_config = OpenAILLMConfig(
        model="gpt-4o-mini", 
        openai_key=OPENAI_API_KEY, 
        stream=False, 
        output_response=False
    )
    optimizer_llm = OpenAILLM(config=optimizer_config)
    
    # 创建工作流
    workflow = create_mass_workflow(executor_llm)
    
    # 创建优化器
    try:
        mass_optimizer = MassOptimiser(
            workflow=workflow,
            optimizer_llm=optimizer_llm,
            max_bootstrapped_demos=2,  # 减少demo数量
            max_labeled_demos=4,
            auto="light",
            eval_rounds=1,
            num_threads=1,  # 减少线程数
            save_path=None,
            max_steps=2  # 减少步数
        )
        print("MASS优化器创建成功")
        return mass_optimizer
    except Exception as e:
        print(f"MASS优化器创建失败: {e}")
        return None


def main():
    """主测试函数"""
    print("开始测试新的MASS优化器...")
    
    # 测试工作流创建
    workflow = test_workflow_creation()
    if workflow is None:
        print("工作流创建失败，退出测试")
        return
    
    # 测试工作流执行
    execution_success = test_workflow_execution(workflow)
    
    # 测试工作流保存和加载
    save_load_success = test_workflow_save_load(workflow)
    
    # 测试MASS优化器创建
    optimizer = test_mass_optimizer_creation()
    
    # 总结测试结果
    print("\n=== 测试总结 ===")
    print(f"工作流创建: {'成功' if workflow else '失败'}")
    print(f"工作流执行: {'成功' if execution_success else '失败'}")
    print(f"工作流保存/加载: {'成功' if save_load_success else '失败'}")
    print(f"MASS优化器创建: {'成功' if optimizer else '失败'}")
    
    if all([workflow, execution_success, save_load_success, optimizer]):
        print("\n所有测试通过！新的MASS优化器工作正常。")
    else:
        print("\n部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()
