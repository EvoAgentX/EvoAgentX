from evoagentx.evaluators.aflow_evaluator import AFlowEvaluator
from ...core.logging import logger
from tqdm import tqdm
import asyncio


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def evaluate_initial_round(self, optimizer, graph_path, directory, validation_n, data):
        # 使用 optimizer 的 graph_utils 来加载图
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = AFlowEvaluator(eval_path=directory, llm=optimizer.optimizer_llm)

        for i in range(validation_n):
            score, avg_cost, total_cost = evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm": optimizer.optimizer_llm},
                directory,
                is_test=False,
            )

            new_data = optimizer.data_utils.create_result_data(optimizer.round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)

        return data

    def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False):
        evaluator = AFlowEvaluator(eval_path=directory, llm=optimizer.optimizer_llm)
        # logger.info(f"evaluator is {evaluator}")
        sum_score = 0

        for _ in tqdm(range(validation_n), desc="Evaluating graph"):
            score, avg_cost, total_cost = evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm": optimizer.optimizer_llm},
                directory,
                is_test=False,
            )
            
            logger.info("{}".format(f"score is {score}, avg_cost is {avg_cost}, total_cost is {total_cost}"), flush=True)

            cur_round = optimizer.round + 1 if initial is False else optimizer.round

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(self.root_path + "/workflows")
            optimizer.data_utils.save_results(result_path, data)

            sum_score += score

        avg_score = sum_score / validation_n
        return avg_score

    def evaluate_graph_test(self, optimizer, directory, is_test=True):
        evaluator = AFlowEvaluator(eval_path=directory, llm=optimizer.optimizer_llm)
        score, avg_cost, total_cost = evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm": optimizer.optimizer_llm, "mode": "dev"},
            directory,
            is_test=is_test,
        )
        return score, avg_cost, total_cost

    async def evaluate_graph_async(self, optimizer, directory, validation_n, data, initial=False):
        evaluator = AFlowEvaluator(eval_path=directory, llm=optimizer.optimizer_llm)
        sum_score = 0
        
        for _ in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate_async(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm": optimizer.optimizer_llm},
                directory,
                is_test=False,
            )
            
            logger.info("{}".format(f"score is {score}, avg_cost is {avg_cost}, total_cost is {total_cost}"), flush=True)

            cur_round = optimizer.round + 1 if initial is False else optimizer.round

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(self.root_path + "/workflows")
            optimizer.data_utils.save_results(result_path, data)
            
            sum_score += score
            
        return sum_score / validation_n


    async def evaluate_graph_test_async(self, optimizer, directory, is_test=True):
        evaluator = AFlowEvaluator(eval_path=directory, llm=optimizer.optimizer_llm)
        score, avg_cost, total_cost = await evaluator.graph_evaluate_async(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm": optimizer.optimizer_llm, "mode": "test"},
            directory,
            is_test=is_test,
        )
        return score, avg_cost, total_cost
