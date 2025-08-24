"""
Layer 3: Workflow Output Quality Evaluation
Evaluates the quality of outputs from successfully executed workflows
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .config import EvaluationConfig
from .utils import (
    retry_with_backoff,
    capture_exception_details,
    save_checkpoint,
    load_checkpoint,
    create_batch_iterator,
    merge_results
)


class OutputQualityEvaluator:
    """Evaluates the quality of workflow outputs using LLM"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.layer_num = 3
    
    def evaluate_output_quality(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of workflow outputs
        
        Args:
            execution_result: Result from Layer 2 execution evaluation
            
        Returns:
            Output quality evaluation results
        """
        start_time = time.time()
        
        try:
            if not execution_result.get('overall_success', False):
                return {
                    'test_id': execution_result.get('test_id'),
                    'workflow_name': execution_result.get('workflow_name'),
                    'evaluation_success': False,
                    'reason': 'Workflow execution failed, no outputs to evaluate',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Extract successful execution results
            successful_executions = [
                exec_result for exec_result in execution_result.get('execution_results', [])
                if exec_result.get('success', False)
            ]
            
            if not successful_executions:
                return {
                    'test_id': execution_result.get('test_id'),
                    'workflow_name': execution_result.get('workflow_name'),
                    'evaluation_success': False,
                    'reason': 'No successful executions found',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Evaluate each successful execution output
            output_evaluations = []
            for i, exec_result in enumerate(successful_executions):
                try:
                    evaluation = self._evaluate_single_output(
                        exec_result.get('output', {}),
                        exec_result.get('input', {}),
                        execution_result
                    )
                    evaluation['execution_index'] = i + 1
                    output_evaluations.append(evaluation)
                except Exception as e:
                    output_evaluations.append({
                        'execution_index': i + 1,
                        'evaluation_success': False,
                        'error': capture_exception_details(e)
                    })
            
            # Aggregate evaluations
            overall_evaluation = self._aggregate_evaluations(output_evaluations)
            
            return {
                'test_id': execution_result.get('test_id'),
                'workflow_name': execution_result.get('workflow_name'),
                'evaluation_success': True,
                'individual_evaluations': output_evaluations,
                'overall_evaluation': overall_evaluation,
                'total_evaluation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'test_id': execution_result.get('test_id'),
                'workflow_name': execution_result.get('workflow_name'),
                'evaluation_success': False,
                'error': capture_exception_details(e),
                'total_evaluation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    @retry_with_backoff(max_retries=3, delay=2.0)
    def _evaluate_single_output(
        self, 
        output: Dict[str, Any], 
        input_data: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single workflow output using LLM
        
        Args:
            output: Workflow output to evaluate
            input_data: Input data that produced this output
            execution_context: Context information about the workflow
            
        Returns:
            Single output evaluation result
        """
        # Get original workflow requirement for context
        workflow_requirement = execution_context.get('workflow_requirement', 'No requirement specified')
        workflow_name = execution_context.get('workflow_name', 'Unknown workflow')
        
        # Evaluate using LLM
        quality_scores = self._evaluate_with_llm(output, input_data, workflow_requirement)
        
        return {
            'evaluation_success': True,
            'input_data': input_data,
            'output_data': output,
            'quality_scores': quality_scores,
            'workflow_requirement': workflow_requirement
        }
    
    def _evaluate_with_llm(
        self, 
        output: Dict[str, Any], 
        input_data: Dict[str, Any], 
        workflow_requirement: str
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate output quality
        
        Args:
            output: Workflow output
            input_data: Input that generated the output
            workflow_requirement: Original workflow requirements
            
        Returns:
            LLM evaluation scores
        """
        evaluation_prompt = f"""
        Please evaluate the following workflow output based on these three criteria:
        
        1. Task Completion (0-10): How well does the output fulfill the original requirements?
        2. Content Consistency (0-10): How logically coherent and internally consistent is the output?
        3. Diversity (0-10): How diverse and non-generic is the output (avoid "plain water" responses)?
        4. Usefulness (0-10): How useful and effective is the output for the specific query?
        
        Original Requirements: {workflow_requirement}
        Input Data: {json.dumps(input_data, indent=2)}
        Generated Output: {json.dumps(output, indent=2)}
        
        Provide scores (0-10) and brief explanations for each criterion.
        """
        
        # This is a placeholder for LLM evaluation
        # In real implementation, you would call your LLM API here
        
        # Simulate LLM response based on output content
        task_completion_score = self._simulate_task_completion_score(output, workflow_requirement)
        consistency_score = self._simulate_consistency_score(output)
        diversity_score = self._simulate_diversity_score(output)
        usefulness_score = self._simulate_usefulness_score(output, input_data)
        
        overall_score = (task_completion_score + consistency_score + diversity_score + usefulness_score) / 4
        
        return {
            'task_completion': {
                'score': task_completion_score,
                'explanation': f'Output addresses {task_completion_score * 10}% of the original requirements'
            },
            'content_consistency': {
                'score': consistency_score,
                'explanation': f'Output shows {"high" if consistency_score > 7 else "moderate" if consistency_score > 5 else "low"} internal consistency'
            },
            'diversity': {
                'score': diversity_score,
                'explanation': f'Output shows {"high" if diversity_score > 7 else "moderate" if diversity_score > 5 else "low"} diversity and specificity'
            },
            'usefulness': {
                'score': usefulness_score,
                'explanation': f'Output is {"highly" if usefulness_score > 7 else "moderately" if usefulness_score > 5 else "minimally"} useful for the given input'
            },
            'overall_score': overall_score,
            'evaluation_prompt': evaluation_prompt
        }
    
    def _simulate_task_completion_score(self, output: Dict[str, Any], requirement: str) -> float:
        """Simulate task completion scoring"""
        # Check if output has expected fields and reasonable values
        if isinstance(output, dict) and len(output) > 0:
            # For content analysis workflow
            if 'categories' in output and 'sentiment_score' in output:
                return 8.5
            elif len(output) >= 2:
                return 7.0
            else:
                return 5.5
        return 3.0
    
    def _simulate_consistency_score(self, output: Dict[str, Any]) -> float:
        """Simulate content consistency scoring"""
        # Check for logical consistency in output structure
        if isinstance(output, dict):
            # Check if all values are properly formatted
            valid_values = 0
            total_values = len(output)
            
            for value in output.values():
                if value is not None and str(value).strip():
                    valid_values += 1
            
            if total_values > 0:
                consistency_ratio = valid_values / total_values
                return 6.0 + (consistency_ratio * 4.0)  # Scale to 6-10 range
        
        return 6.0
    
    def _simulate_diversity_score(self, output: Dict[str, Any]) -> float:
        """Simulate diversity scoring"""
        # Check for specificity and non-generic content
        if isinstance(output, dict):
            # Count specific/detailed values vs generic ones
            specific_content = 0
            total_content = 0
            
            for key, value in output.items():
                total_content += 1
                value_str = str(value).lower()
                
                # Check for specific, non-generic content
                if (len(value_str) > 10 and 
                    'sample' not in value_str and 
                    'default' not in value_str and
                    'generic' not in value_str):
                    specific_content += 1
            
            if total_content > 0:
                diversity_ratio = specific_content / total_content
                return 5.0 + (diversity_ratio * 5.0)  # Scale to 5-10 range
        
        return 5.0
    
    def _simulate_usefulness_score(self, output: Dict[str, Any], input_data: Dict[str, Any]) -> float:
        """Simulate usefulness scoring"""
        # Check if output appears to be related to input
        if isinstance(output, dict) and isinstance(input_data, dict):
            # For content analysis example
            if ('article_content' in input_data and 
                any(key in output for key in ['categories', 'topics', 'sentiment', 'summary'])):
                return 8.0
            elif len(output) > 0:
                return 6.5
        
        return 4.0
    
    def _aggregate_evaluations(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple output evaluations into overall assessment
        
        Args:
            evaluations: List of individual output evaluations
            
        Returns:
            Aggregated evaluation results
        """
        successful_evaluations = [
            eval_result for eval_result in evaluations
            if eval_result.get('evaluation_success', False)
        ]
        
        if not successful_evaluations:
            return {
                'aggregation_success': False,
                'reason': 'No successful individual evaluations to aggregate'
            }
        
        # Calculate average scores across all criteria
        criteria = ['task_completion', 'content_consistency', 'diversity', 'usefulness']
        aggregated_scores = {}
        
        for criterion in criteria:
            scores = []
            explanations = []
            
            for evaluation in successful_evaluations:
                quality_scores = evaluation.get('quality_scores', {})
                if criterion in quality_scores:
                    scores.append(quality_scores[criterion]['score'])
                    explanations.append(quality_scores[criterion]['explanation'])
            
            if scores:
                aggregated_scores[criterion] = {
                    'average_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'score_variance': max(scores) - min(scores) if len(scores) > 1 else 0,
                    'sample_explanations': explanations[:2]  # Keep first 2 explanations as examples
                }
        
        # Calculate overall aggregated score
        if aggregated_scores:
            overall_average = sum(
                scores['average_score'] for scores in aggregated_scores.values()
            ) / len(aggregated_scores)
        else:
            overall_average = 0
        
        return {
            'aggregation_success': True,
            'total_evaluations': len(evaluations),
            'successful_evaluations': len(successful_evaluations),
            'aggregated_scores': aggregated_scores,
            'overall_average_score': overall_average,
            'quality_assessment': self._get_quality_assessment(overall_average)
        }
    
    def _get_quality_assessment(self, overall_score: float) -> str:
        """Get qualitative assessment based on overall score"""
        if overall_score >= 8.0:
            return "Excellent"
        elif overall_score >= 7.0:
            return "Good"
        elif overall_score >= 6.0:
            return "Satisfactory"
        elif overall_score >= 5.0:
            return "Below Average"
        else:
            return "Poor"


def evaluate_outputs_batch(
    execution_results_batch: List[Dict[str, Any]], 
    config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Evaluate output quality for a batch of execution results
    
    Args:
        execution_results_batch: Batch of Layer 2 execution results
        config: Evaluation configuration
        
    Returns:
        Batch output quality evaluation results
    """
    evaluator = OutputQualityEvaluator(config)
    results = []
    errors = []
    
    start_time = time.time()
    
    for execution_result in execution_results_batch:
        try:
            result = evaluator.evaluate_output_quality(execution_result)
            results.append(result)
        except Exception as e:
            error_info = capture_exception_details(e)
            error_info['test_id'] = execution_result.get('test_id', 'unknown')
            errors.append(error_info)
    
    evaluation_time = time.time() - start_time
    
    return {
        'test_results': results,
        'errors': errors,
        'summary': {
            'total_tests': len(execution_results_batch),
            'successful': len(results),
            'failed': len(errors),
            'execution_time': evaluation_time
        }
    }


def run_layer_3_evaluation(
    layer_2_results: Dict[str, Any], 
    config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Run Layer 3 output quality evaluation with multiprocessing
    
    Args:
        layer_2_results: Results from Layer 2 execution evaluation
        config: Evaluation configuration
        
    Returns:
        Complete output quality evaluation results
    """
    print(f"Starting Layer 3 evaluation...")
    
    # Extract successful execution results from Layer 2
    successful_executions = [
        result for result in layer_2_results.get('test_results', [])
        if result.get('overall_success', False)
    ]
    
    if not successful_executions:
        print("No successful workflow executions found from Layer 2 to evaluate")
        return {
            'layer': 3,
            'test_results': [],
            'errors': [],
            'summary': {'total_tests': 0, 'successful': 0, 'failed': 0},
            'message': 'No successful executions available for output quality evaluation'
        }
    
    print(f"Found {len(successful_executions)} successful executions to evaluate...")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(config.checkpoint_dir, 3)
    if checkpoint:
        print("Found existing checkpoint for Layer 3, resuming from last state...")
        completed_ids = {result['test_id'] for result in checkpoint.get('test_results', [])}
        successful_executions = [
            execution for execution in successful_executions 
            if execution.get('test_id') not in completed_ids
        ]
        print(f"Resuming with {len(successful_executions)} remaining executions...")
    else:
        checkpoint = {'test_results': [], 'errors': [], 'summary': {}}
    
    if not successful_executions:
        print("All executions already evaluated for Layer 3")
        return checkpoint
    
    # Create batches for parallel processing
    batches = create_batch_iterator(successful_executions, config.batch_size)
    all_results = []
    
    # Determine optimal number of processes
    num_processes = min(config.max_processes, len(batches))
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for parallel evaluation
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(evaluate_outputs_batch, batch, config): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_result = future.result()
                all_results.append(batch_result)
                print(f"Completed evaluation batch {batch_idx + 1}/{len(batches)}")
                
                # Save intermediate checkpoint
                if len(all_results) % 3 == 0:  # Save every 3 batches
                    intermediate_results = merge_results([checkpoint] + all_results)
                    save_checkpoint(config.checkpoint_dir, 3, intermediate_results)
                    
            except Exception as e:
                print(f"Evaluation batch {batch_idx + 1} failed: {e}")
                error_info = capture_exception_details(e)
                all_results.append({
                    'test_results': [],
                    'errors': [error_info],
                    'summary': {'total_tests': len(batches[batch_idx]), 'successful': 0, 'failed': 1}
                })
    
    # Merge all results
    final_results = merge_results([checkpoint] + all_results)
    final_results['layer'] = 3
    final_results['total_execution_time'] = time.time() - start_time
    final_results['timestamp'] = datetime.now().isoformat()
    
    # Save final checkpoint
    save_checkpoint(config.checkpoint_dir, 3, final_results)
    
    # Save final results
    results_file = os.path.join(config.results_dir, 'layer_3_output_evaluation.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"Layer 3 evaluation completed in {final_results['total_execution_time']:.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    return final_results
