import numpy as np
import random
import json
import os
from typing import Any, Optional, List, Dict
from pydantic import Field

from evoagentx.core.module import BaseModule
from evoagentx.models.base_model import BaseLLM
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.optimizers import MiproOptimizer
from evoagentx.utils.mipro_utils.register_utils import MiproRegistry

# 导入新的blocks
from .blocks import (
    Predictor, create_predictor_agent,
    Aggregate, create_aggregate,
    Debate, create_debate_agent,
    Reflect, create_reflect_agent,
    Summarize, create_summarize_agent,
    Execute, create_execute_agent
)


class MassBlock:
    """MASS Block基类，用于统一管理各个block的属性和方法"""
    
    def __init__(self, block_instance, name: str, search_space: List[int] = None):
        self.block = block_instance
        self.name = name
        self.n = 0  # 当前激活的n值
        self.activate = False  # 是否激活
        self.search_space = search_space or [1, 3, 5, 7, 9]
        self.influence_score = 0.0  # 影响力分数
        
    def execute(self, *args, **kwargs):
        """执行block"""
        return self.block(*args, **kwargs)
    
    def get_registry(self) -> List[str]:
        """获取需要注册的参数"""
        if hasattr(self.block, 'get_registry'):
            return self.block.get_registry()
        # 对于CustomizeAgent，注册prompt参数
        if hasattr(self.block, 'agent') and hasattr(self.block.agent, 'prompt'):
            return [f"{self.name}.agent.prompt"]
        return [f"{self.name}.prompt"]
    
    def save(self, path: str):
        """保存block配置"""
        if hasattr(self.block, 'save'):
            self.block.save(path)
    
    def load(self, path: str):
        """加载block配置"""
        if hasattr(self.block, 'load'):
            self.block.load(path)


class MassWorkflow:
    """基于新blocks的MASS工作流，匹配原始WorkFlow的逻辑"""
    
    def __init__(self, blocks: List[MassBlock]):
        self.blocks = blocks
        # 按照原始实现的顺序：summarizer, aggregater, reflector, debater, executer
        self.summarizer = blocks[0] if len(blocks) > 0 else None
        self.aggregater = blocks[1] if len(blocks) > 1 else None
        self.reflector = blocks[2] if len(blocks) > 2 else None
        self.debater = blocks[3] if len(blocks) > 3 else None
        self.executer = blocks[4] if len(blocks) > 4 else None
    
    def get_state(self):
        """获取当前工作流状态"""
        return {
            'blocks': [
                {
                    'name': block.name,
                    'n': block.n,
                    'activate': block.activate,
                    'search_space': block.search_space,
                    'influence_score': block.influence_score,
                    'prompt': block.block.agent.prompt if hasattr(block.block, 'agent') else None
                }
                for block in self.blocks
            ]
        }
    
    def set_state(self, state):
        """设置工作流状态"""
        for i, block_state in enumerate(state['blocks']):
            if i < len(self.blocks):
                block = self.blocks[i]
                block.n = block_state['n']
                block.activate = block_state['activate']
                block.search_space = block_state['search_space']
                block.influence_score = block_state['influence_score']
                if block_state['prompt'] and hasattr(block.block, 'agent'):
                    block.block.agent.prompt = block_state['prompt']
    
    def __call__(self, problem: str, **kwargs):
        context = kwargs.get("context", None)
        testcases = kwargs.get("testcases", None)
        
        # Step 1: 获取总结的上下文（匹配原始逻辑）
        if self.summarizer and self.summarizer.n > 0:
            context = self.summarizer.execute(problem, context=context)

        # Step 2: 生成候选解决方案（匹配原始逻辑）
        if self.debater and self.debater.n > 0:
            candidate_solutions = self.aggregater.execute(problem, context=context)
        else:
            candidate_solutions = [self.aggregater.execute(problem)]

        # Step 3: 对每个候选方案进行反思优化（匹配原始逻辑）
        if self.reflector and self.reflector.n > 0:
            for i in range(len(candidate_solutions)):
                if self.executer and self.executer.n > 0:
                    self.executer.n = self.reflector.n
                    refined_answer = self.executer.execute(problem, candidate_solutions[i], testcases=testcases)
                else:
                    refined_answer = self.reflector.execute(problem, candidate_solutions[i], context=context)
                candidate_solutions[i] = refined_answer

        # Step 4: 通过辩论选择最佳答案（匹配原始逻辑）
        if self.debater and self.debater.n > 0:
            final_answer = self.debater.execute(problem, candidate_solutions, context=context)
        else:
            final_answer = candidate_solutions[0]

        return final_answer, {
            "problem": problem, 
            "context": context, 
            "testcases": testcases,
            "answer": final_answer
        }
    
    def save(self, path: str):
        """保存工作流配置，匹配原始实现的保存逻辑"""
        params = {
            "summarizer": {
                "summarizer": self.summarizer.block.agent.prompt if self.summarizer else "",
                "predictor": self.summarizer.block.agent.prompt if self.summarizer else "",
            },
            "aggregater": {
                "predictor": self.aggregater.block.agent.prompt if self.aggregater else "",
            },
            "reflector": {
                "reflector": self.reflector.block.agent.prompt if self.reflector else "",
                "refiner": self.reflector.block.agent.prompt if self.reflector else "",
            },
            "debater": {
                "debater": self.debater.block.agent.prompt if self.debater else "",
                "predictor": self.debater.block.agent.prompt if self.debater else "",
            },
            "executer": {
                "predictor": self.executer.block.agent.prompt if self.executer else "",
                "codereflector": self.executer.block.agent.prompt if self.executer else "",
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """加载工作流配置，匹配原始实现的加载逻辑"""
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
            
        if self.summarizer:
            self.summarizer.block.agent.prompt = params["summarizer"]["summarizer"]
        if self.aggregater:
            self.aggregater.block.agent.prompt = params["aggregater"]["predictor"]
        if self.reflector:
            self.reflector.block.agent.prompt = params["reflector"]["reflector"]
        if self.debater:
            self.debater.block.agent.prompt = params["debater"]["debater"]
        if self.executer:
            self.executer.block.agent.prompt = params["executer"]["predictor"]


class MassOptimiser(BaseModule):
    """基于新blocks的MASS优化器，匹配原始优化流程"""
    
    workflow: MassWorkflow = Field(default=None, description="The workflow to optimize.")
    optimizer_llm: Optional[BaseLLM] = Field(default=None, description="The LLM to use for optimization.")
    max_bootstrapped_demos: int = Field(default=5, description="The number of bootstrapped demos to use for optimization.")
    max_labeled_demos: int = Field(default=16, description="The number of labeled demos to use for optimization.")
    auto: str = Field(default="medium", description="The auto mode to use for optimization.")
    eval_rounds: int = Field(default=1, description="The number of rounds to evaluate the workflow.")
    num_threads: int = Field(default=5, description="The number of threads")
    save_path: Optional[str] = Field(default=None, description="Directory for saving logs, None means no saving")
    max_steps: int = Field(default=10, description="Maximum optimization steps")

    def init_module(self, **kwargs):
        self.rng = None
        if self.optimizer_llm is None:
            raise ValueError("Optimizer llm is required")

    def optimize(self, 
                 *,
                 benchmark: Benchmark,
                 softmax_temperature: float = 1.0,
                 agent_budget: int = 10):
        """优化工作流，匹配原始优化流程"""
        self.benchmark = benchmark
        selection_probability = self._softmax_with_temperature(softmax_temperature)
        best_score = 0
        best_workflow = None
        
        for step in range(self.max_steps):
            print(f"Optimization step {step + 1}/{self.max_steps}")
            
            # 创建registry
            registry = MiproRegistry()
            
            # 随机选择blocks（匹配原始逻辑）
            u = np.random.uniform(0, 1, size=selection_probability.shape)
            total = 0
            
            for ui, pi, block in zip(u, selection_probability, self.workflow.blocks):
                if ui <= pi:
                    # 选择第一个搜索空间值
                    block.n = block.search_space[0]
                    total += block.n
                else:
                    # 随机选择搜索空间值
                    space = block.search_space
                    idx = random.randint(0, len(space) - 1)
                    block.n = space[idx]
                    total += block.n
                
                # 激活block
                if block.n > 0:
                    block.activate = True
                    
                    # 注册需要优化的参数（匹配原始注册逻辑）
                    for register in block.get_registry():
                        print(f"Registering {register} for {block.name}")
                        registry.track(block, register, input_names=['problem'], output_names=['answer'])
                else:
                    block.activate = False
            
            # 检查预算限制
            if total > agent_budget:
                print(f"Budget exceeded: {total} > {agent_budget}, skipping this step")
                continue
            
            # 创建优化器
            optimizer = MiproOptimizer(
                registry=registry,
                program=self.workflow,
                optimizer_llm=self.optimizer_llm,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                num_threads=self.num_threads,
                eval_rounds=self.eval_rounds,
                auto=self.auto,
                save_path=self.save_path
            )
            
            # 评估当前配置
            score = optimizer.evaluate(dataset=self.benchmark, eval_mode="test")
            print(f"Step {step + 1} score: {score}")
            
            # 更新最佳结果
            if score > best_score:
                best_score = score
                # 保存当前工作流状态
                best_workflow_state = self.workflow.get_state()
                print(f"New best score: {best_score}")
        
        # 使用最佳配置进行最终优化
        if 'best_workflow_state' in locals():
            # 恢复最佳工作流状态
            self.workflow.set_state(best_workflow_state)
            
            final_optimizer = MiproOptimizer(
                registry=registry,
                program=self.workflow,
                optimizer_llm=self.optimizer_llm,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                num_threads=self.num_threads,
                eval_rounds=self.eval_rounds,
                auto=self.auto,
                save_path=self.save_path
            )
            
            final_optimizer.optimize(dataset=self.benchmark)
            return final_optimizer.restore_best_program()
        
        return self.workflow

    def _softmax_with_temperature(self, temperature):        
        """计算softmax概率分布"""
        logits = []
        for block in self.workflow.blocks:
            logits.append(block.influence_score)

        logits = np.array(logits, dtype=np.float32)
        logits = logits / temperature
        exps = np.exp(logits - np.max(logits))

        return exps / np.sum(exps)


def create_mass_workflow(executor_llm: BaseLLM) -> MassWorkflow:
    """创建MASS工作流，按照原始实现的顺序"""
    # 创建各个blocks，按照原始实现的顺序
    predictor = create_predictor_agent(executor_llm)
    aggregate = create_aggregate(predictor, n=3)
    debate = create_debate_agent(executor_llm)
    reflect = create_reflect_agent(executor_llm)
    summarize = create_summarize_agent(executor_llm)
    execute = create_execute_agent(executor_llm)
    
    # 包装成MassBlock，按照原始顺序：summarizer, aggregater, reflector, debater, executer
    blocks = [
        MassBlock(summarize, "summarizer"),  # 对应原始 self.summarizer
        MassBlock(aggregate, "aggregater"),  # 对应原始 self.aggregater
        MassBlock(reflect, "reflector"),     # 对应原始 self.reflector
        MassBlock(debate, "debater"),        # 对应原始 self.debater
        MassBlock(execute, "executer")       # 对应原始 self.executer
    ]
    
    return MassWorkflow(blocks)


def optimize_predictor(predictor: Predictor, optimizer_llm: BaseLLM, benchmark: Benchmark):
    """Step 0: 优化单独的predictor（匹配原始实现）"""
    registry = MiproRegistry()
    registry.track(predictor, "agent.prompt", input_names=['problem'], output_names=['reasoning', 'answer'])
    
    optimizer = MiproOptimizer(
        registry=registry,
        program=predictor,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=5,
        max_labeled_demos=16,
        num_threads=5,
        eval_rounds=1,
        auto="medium",
        save_path="examples/mass/predictor"
    )
    
    optimizer.optimize(dataset=benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    optimized_predictor = optimizer.restore_best_program()
    
    return score, optimized_predictor


def optimize_summarizer(optimized_predictor: Predictor, executor_llm: BaseLLM, optimizer_llm: BaseLLM, 
                       benchmark: Benchmark, predictor_score: float):
    """Step 1: 优化summarizer block（匹配原始实现）"""
    block = create_summarize_agent(executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "agent.prompt", input_names=['problem'], output_names=['summary'])
    
    optimizer = MiproOptimizer(
        registry=registry,
        program=block,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=5,
        max_labeled_demos=16,
        num_threads=5,
        eval_rounds=1,
        auto="medium",
        save_path="examples/mass/summarizer"
    )
    
    optimizer.optimize(dataset=benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    mass_block = MassBlock(optimized_block, "summarizer")
    mass_block.influence_score = influence
    
    return mass_block


def optimize_aggregator(optimized_predictor: Predictor, optimizer_llm: BaseLLM, 
                       benchmark: Benchmark, predictor_score: float):
    """Step 1: 优化aggregator block（匹配原始实现）"""
    block = create_aggregate(optimized_predictor, n=3)
    
    registry = MiproRegistry()
    registry.track(block, "predictor.agent.prompt", input_names=['problem'], output_names=['reasoning', 'answer'])
    
    optimizer = MiproOptimizer(
        registry=registry,
        program=block,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=5,
        max_labeled_demos=16,
        num_threads=5,
        eval_rounds=1,
        auto="medium",
        save_path="examples/mass/aggregator"
    )
    
    optimizer.optimize(dataset=benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    mass_block = MassBlock(optimized_block, "aggregater")
    mass_block.influence_score = influence
    
    return mass_block


def optimize_reflector(optimized_predictor: Predictor, executor_llm: BaseLLM, optimizer_llm: BaseLLM, 
                      benchmark: Benchmark, predictor_score: float):
    """Step 1: 优化reflector block（匹配原始实现）"""
    block = create_reflect_agent(executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "agent.prompt", input_names=['problem'], output_names=['reasoning', 'feedback', 'correctness'])
    
    optimizer = MiproOptimizer(
        registry=registry,
        program=block,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=5,
        max_labeled_demos=16,
        num_threads=5,
        eval_rounds=1,
        auto="medium",
        save_path="examples/mass/reflector"
    )
    
    optimizer.optimize(dataset=benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    mass_block = MassBlock(optimized_block, "reflector")
    mass_block.influence_score = influence
    
    return mass_block


def optimize_debater(optimized_predictor: Predictor, executor_llm: BaseLLM, optimizer_llm: BaseLLM, 
                    benchmark: Benchmark, predictor_score: float):
    """Step 1: 优化debater block（匹配原始实现）"""
    block = create_debate_agent(executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "agent.prompt", input_names=['problem'], output_names=['reasoning', 'index', 'answer'])
    
    optimizer = MiproOptimizer(
        registry=registry,
        program=block,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=5,
        max_labeled_demos=16,
        num_threads=5,
        eval_rounds=1,
        auto="medium",
        save_path="examples/mass/debater"
    )
    
    optimizer.optimize(dataset=benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    mass_block = MassBlock(optimized_block, "debater")
    mass_block.influence_score = influence
    
    return mass_block


def optimize_executer(optimized_predictor: Predictor, executor_llm: BaseLLM, optimizer_llm: BaseLLM, 
                     benchmark: Benchmark, predictor_score: float):
    """Step 1: 优化executer block（匹配原始实现）"""
    block = create_execute_agent(executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "agent.prompt", input_names=['problem'], output_names=['traceback', 'correctness'])
    
    optimizer = MiproOptimizer(
        registry=registry,
        program=block,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=5,
        max_labeled_demos=16,
        num_threads=5,
        eval_rounds=1,
        auto="medium",
        save_path="examples/mass/executer"
    )
    
    optimizer.optimize(dataset=benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    mass_block = MassBlock(optimized_block, "executer")
    mass_block.influence_score = influence
    
    return mass_block


def run_full_optimization(executor_llm: BaseLLM, optimizer_llm: BaseLLM, benchmark: Benchmark):
    """运行完整的优化流程，匹配原始main函数的逻辑"""
    
    # Step 0: 优化 Predictor
    print("Step 0: 优化 Predictor...")
    predictor = create_predictor_agent(executor_llm)
    predictor_score, optimized_predictor = optimize_predictor(predictor, optimizer_llm, benchmark)
    print(f"Predictor优化完成，分数: {predictor_score}")

    # Step 1: 逐个优化每个block
    print("\nStep 1: 逐个优化每个block...")
    
    print("优化 summarizer...")
    optimized_summarizer = optimize_summarizer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 aggregator...")
    optimized_aggregator = optimize_aggregator(optimized_predictor, optimizer_llm, benchmark, predictor_score)
    
    print("优化 reflector...")
    optimized_reflector = optimize_reflector(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 debater...")
    optimized_debater = optimize_debater(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 executer...")
    optimized_executer = optimize_executer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)

    # Step 2: 构建最终工作流
    print("\nStep 2: 构建最终工作流...")
    block_workflow = MassWorkflow([
        optimized_summarizer,
        optimized_aggregator,
        optimized_reflector,
        optimized_debater,
        optimized_executer
    ])

    # Step 3: 优化完整工作流（包括n和prompt）
    print("\nStep 3: 优化完整工作流...")
    mass = MassOptimiser(
        workflow=block_workflow,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=5,
        max_labeled_demos=16,
        auto="medium",
        eval_rounds=1,
        num_threads=5,
        save_path="examples/mass/mass_optimization",
        max_steps=10
    )

    best_program = mass.optimize(benchmark=benchmark)
    
    # 保存最佳工作流配置
    block_workflow.save("examples/mass/best_workflow_config.json")
    print("Best workflow config saved to examples/mass/best_workflow_config.json")
    
    return best_program
