"""
Layer 1: Workflow Structure Evaluation
Evaluates the structural integrity and logical coherence of generated workflow JSON files
call hierarchy: run_layer_1_evaluation -> evaluate_structure_batch -> async_evaluate_structure_batch -> evaluate_workflow_structure
"""
import asyncio
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any

from .base_evaluator import BaseEvaluator
from .config import EvaluationConfig
from .utils import (
    async_retry_with_backoff,
    capture_exception_details, 
    save_checkpoint, 
    load_checkpoint,
    create_batch_iterator,
    merge_results
)


class StructureEvaluator(BaseEvaluator):
    """Evaluates workflow structure using LLM"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.layer_num = 1
    
    @async_retry_with_backoff(max_retries=3, delay=2.0)
    async def evaluate_workflow_structure(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate workflow structure using LLM
        
        Args:
            workflow_data: Workflow data to evaluate
            
        Returns:
            Evaluation results
        """
        # Import here to avoid circular dependencies
        from evoagentx.workflow.workflow_generator import WorkFlowGenerator
        from evoagentx.core.base_config import Parameter
        from .exceptions import WorkFlowGenerationFailed
        
        try:
            # Generate workflow from test data
            generator = WorkFlowGenerator(
                llm=self.llm,
                tools=self.tools,
            ) # not support for prebuilt agents and workflow examples now
            
            # Convert test data to workflow parameters
            workflow_inputs = [
                Parameter(**inp) for inp in workflow_data.get('workflow_inputs', [])
            ]
            workflow_outputs = [
                Parameter(**out) for out in workflow_data.get('workflow_outputs', [])
            ]
            
            # Generate workflow
            try:
                workflow_graph = await asyncio.to_thread(
                    generator.generate_workflow,
                    goal=workflow_data.get('workflow_requirement', ''),
                    workflow_inputs=workflow_inputs,
                    workflow_outputs=workflow_outputs
                )
            except Exception as e:
                import traceback
                print("Underlying exception while generating workflow:")
                traceback.print_exc()
                raise WorkFlowGenerationFailed(f"Failed to generate workflow for {workflow_data.get('workflow_id')}: {e}")
            
            # Serialize workflow to JSON
            workflow_json_str = workflow_graph.to_json()
            
            # Evaluate structure using LLM
            structure_score = await self._evaluate_with_llm(workflow_json_str, workflow_data)
            
            return {
                'test_id': workflow_data.get('workflow_id'),
                'workflow_name': workflow_data.get('workflow_name'),
                'success': True,
                'workflow_json': workflow_json_str,
                'structure_evaluation': structure_score,
                'execution_time': time.time(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'test_id': workflow_data.get('workflow_id'),
                'workflow_name': workflow_data.get('workflow_name'),
                'success': False,
                'error': capture_exception_details(e),
                'execution_time': time.time(),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _evaluate_with_llm(self, workflow_json_str: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to evaluate workflow structure
        
        Args:
            workflow_dict: Generated workflow dictionary
            original_data: Original test data
            
        Returns:
            LLM evaluation results
        """
        # This is a placeholder for LLM evaluation
        # In real implementation, you would call your LLM API here
        
        evaluation_prompt = f"""
        Please evaluate the following generated workflow structure based on these criteria:
        
        1. Structural Integrity (0-10): Are all required components present?
        2. Input/Output Matching (0-10): Do inputs and outputs align properly?
        3. Task Decomposition Logic (0-10): Is the task breakdown logical and coherent?
        
        Original Requirements: {original_data.get('workflow_requirement', '')}
        Required workflow inputs: 
        ```json
        {json.dumps(original_data.get('workflow_inputs', []), indent=2)}
        ```

        Required workflow outputs: 
        ```json
        {json.dumps(original_data.get('workflow_outputs', []), indent=2)}
        ```

        Generated Workflow: 
        ```json
        {workflow_json_str}
        ```

        Provide scores and brief explanations for each criterion.
        """
        
        output = await self.llm.single_generate_async(
            messages=[
                {"role": "system", "content": evaluation_prompt},
                {"role": "user", "content": "Evaluate the workflow structure"}
            ],
            response_format={"type": "json_object"}
        )

        return output


async def async_evaluate_structure_batch(
    test_batch: List[Dict[str, Any]], 
    config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Asynchronously evaluate a batch of workflows in a single process
    
    Args:
        test_batch: Batch of test data
        config: Evaluation configuration
        
    Returns:
        Batch evaluation results
    """
    evaluator = StructureEvaluator(config)
    results = []
    errors = []
    
    start_time = time.time()
    
    # Create async tasks for all test data in the batch
    async def evaluate_single_test(test_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await evaluator.evaluate_workflow_structure(test_data)
            if result['success']:
                return {'type': 'result', 'data': result}
            else:
                return {'type': 'error', 'data': result}
        except Exception as e:
            error_info = capture_exception_details(e)
            error_info['test_id'] = test_data.get('workflow_id', 'unknown')
            return {'type': 'error', 'data': error_info}
    
    # Run all evaluations concurrently within this batch
    tasks = [evaluate_single_test(test_data) for test_data in test_batch]
    
    # Use asyncio.gather to run all tasks concurrently
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and separate successful evaluations from errors
    for result in batch_results:
        if isinstance(result, Exception):
            # Handle any exceptions that weren't caught in evaluate_single_test
            error_info = capture_exception_details(result)
            error_info['test_id'] = 'unknown'
            errors.append(error_info)
        elif result['type'] == 'result':
            results.append(result['data'])
        elif result['type'] == 'error':
            errors.append(result['data'])
    
    execution_time = time.time() - start_time
    
    return {
        'test_results': results,
        'errors': errors,
        'summary': {
            'total_tests': len(test_batch),
            'successful': len(results),
            'failed': len(errors),
            'execution_time': execution_time
        }
    }


def evaluate_structure_batch(
    test_batch: List[Dict[str, Any]], 
    config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Synchronous wrapper for async batch evaluation (for multiprocessing compatibility)
    
    Args:
        test_batch: Batch of test data
        config: Evaluation configuration
        
    Returns:
        Batch evaluation results
    """
    # Create new event loop for this process, beca
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(async_evaluate_structure_batch(test_batch, config))
    finally:
        loop.close()


def run_layer_1_evaluation(test_data: List[Dict[str, Any]], config: EvaluationConfig) -> Dict[str, Any]:
    """
    Run Layer 1 structure evaluation with multiprocessing
    
    Args:
        test_data: List of test data
        config: Evaluation configuration
        
    Returns:
        Complete evaluation results
    """
    print(f"Starting Layer 1 evaluation with {len(test_data)} test cases...")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(config.checkpoint_dir, 1)
    if checkpoint:
        print("Found existing checkpoint for Layer 1, resuming from last state...")
        completed_ids = {result['test_id'] for result in checkpoint.get('test_results', [])}
        test_data = [test for test in test_data if test.get('workflow_id') not in completed_ids]
        print(f"Resuming with {len(test_data)} remaining test cases...")
    else:
        checkpoint = {'test_results': [], 'errors': [], 'summary': {}}
    
    if not test_data:
        print("All test cases already completed for Layer 1")
        return checkpoint
    
    # Create batches for parallel processing
    batches = create_batch_iterator(test_data, config.batch_size)
    all_results = []
    
    # Determine optimal number of processes
    num_processes = min(config.max_processes, len(batches))
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all batches, each batch will be processed in a separate process
        future_to_batch = {
            executor.submit(evaluate_structure_batch, batch, config): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_result = future.result()
                all_results.append(batch_result)
                print(f"Completed batch {batch_idx + 1}/{len(batches)}")
                
                # Save intermediate checkpoint
                if len(all_results) % 5 == 0:  # Save every 5 batches
                    intermediate_results = merge_results([checkpoint] + all_results)
                    save_checkpoint(config.checkpoint_dir, 1, intermediate_results)
                    
            except Exception as e:
                print(f"Batch {batch_idx + 1} failed: {e}")
                error_info = capture_exception_details(e)
                all_results.append({
                    'test_results': [],
                    'errors': [error_info],
                    'summary': {'total_tests': len(batches[batch_idx]), 'successful': 0, 'failed': 1}
                })
    
    # Merge all results
    final_results = merge_results([checkpoint] + all_results)
    final_results['layer'] = 1
    final_results['total_execution_time'] = time.time() - start_time
    final_results['timestamp'] = datetime.now().isoformat()
    
    # Save final checkpoint
    save_checkpoint(config.checkpoint_dir, 1, final_results)
    
    # Save final results
    results_file = os.path.join(config.results_dir, 'layer_1_structure_evaluation.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"Layer 1 evaluation completed in {final_results['total_execution_time']:.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    return final_results
