"""
Main evaluation script that orchestrates all three evaluation layers
"""
import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

from .config import EvaluationConfig
from .utils import load_test_data, get_optimal_process_count
from .layer_1_structure_eval import run_layer_1_evaluation
from .layer_2_execution_eval import run_layer_2_evaluation  
from .layer_3_output_eval import run_layer_3_evaluation


class WorkflowEvaluationPipeline:
    """Main evaluation pipeline coordinator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.start_time = None
        self.results = {}
    
    def run_complete_evaluation(self, test_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run complete three-layer evaluation pipeline
        
        Args:
            test_data: Optional test data, if not provided will load from config directory
            
        Returns:
            Complete evaluation results
        """
        self.start_time = time.time()
        
        print("=" * 60)
        print("WORKFLOW EVALUATION PIPELINE")
        print("=" * 60)
        
        # Load test data if not provided
        if test_data is None:
            print(f"Loading test data from: {self.config.eval_data_dir}")
            test_data = load_test_data(self.config.eval_data_dir)
        
        if not test_data:
            raise ValueError("No test data found or provided")
        
        print(f"Loaded {len(test_data)} test cases")
        print(f"Layers to run: {self.config.layers_to_run}")
        print(f"Max processes: {self.config.max_processes}")
        print(f"Batch size: {self.config.batch_size}")
        print()
        
        # Run evaluation layers
        layer_1_results = None
        layer_2_results = None
        layer_3_results = None
        
        try:
            # Layer 1: Structure Evaluation
            if 1 in self.config.layers_to_run:
                print("Running Layer 1: Structure Evaluation")
                print("-" * 40)
                layer_1_results = run_layer_1_evaluation(test_data, self.config)
                self.results['layer_1'] = layer_1_results
                self._print_layer_summary(1, layer_1_results)
                print()
            
            # Layer 2: Execution Evaluation
            if 2 in self.config.layers_to_run:
                if layer_1_results is None:
                    print("Loading Layer 1 results for Layer 2 evaluation...")
                    layer_1_results = self._load_layer_results(1)
                
                if layer_1_results:
                    print("Running Layer 2: Execution Evaluation")
                    print("-" * 40)
                    layer_2_results = run_layer_2_evaluation(layer_1_results, self.config)
                    self.results['layer_2'] = layer_2_results
                    self._print_layer_summary(2, layer_2_results)
                    print()
                else:
                    print("Warning: No Layer 1 results available for Layer 2 evaluation")
            
            # Layer 3: Output Quality Evaluation  
            if 3 in self.config.layers_to_run:
                if layer_2_results is None:
                    print("Loading Layer 2 results for Layer 3 evaluation...")
                    layer_2_results = self._load_layer_results(2)
                
                if layer_2_results:
                    print("Running Layer 3: Output Quality Evaluation")
                    print("-" * 40)
                    layer_3_results = run_layer_3_evaluation(layer_2_results, self.config)
                    self.results['layer_3'] = layer_3_results
                    self._print_layer_summary(3, layer_3_results)
                    print()
                else:
                    print("Warning: No Layer 2 results available for Layer 3 evaluation")
            
            # Generate final comprehensive report
            try:
                final_report = self._generate_final_report()
            except Exception as e:
                print(f"Error generating final report: {e}")
                print("Final report will not be generated.")
                final_report = None
            
            return final_report
            
        except Exception as e:
            print(f"Evaluation pipeline failed: {e}")
            raise
        
        finally:
            total_time = time.time() - self.start_time
            print(f"Pipeline completed in {total_time:.2f} seconds")
    
    def _load_layer_results(self, layer: int) -> Optional[Dict[str, Any]]:
        """Load results from a specific layer"""
        results_file = os.path.join(
            self.config.results_dir, 
            f'layer_{layer}_{"structure" if layer == 1 else "execution" if layer == 2 else "output"}_evaluation.json'
        )
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading Layer {layer} results: {e}")
        
        return None
    
    def _print_layer_summary(self, layer: int, results: Dict[str, Any]):
        """Print summary statistics for a layer"""
        if not results:
            print(f"Layer {layer}: No results")
            return
        
        summary = results.get('summary', {})
        total_tests = summary.get('total_tests', 0)
        successful = summary.get('successful', 0)
        failed = summary.get('failed', 0)
        execution_time = summary.get('execution_time', 0)
        
        success_rate = (successful / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Execution time: {execution_time:.2f}s")
        
        # Layer-specific information
        if layer == 2:
            # Show execution statistics
            test_results = results.get('test_results', [])
            if test_results:
                avg_success_rate = sum(
                    result.get('statistics', {}).get('success_rate', 0) 
                    for result in test_results if result.get('overall_success', False)
                ) / len([r for r in test_results if r.get('overall_success', False)])
                print(f"Average workflow success rate: {avg_success_rate * 100:.1f}%")
        
        elif layer == 3:
            # Show quality statistics
            test_results = results.get('test_results', [])
            if test_results:
                quality_scores = []
                for result in test_results:
                    if result.get('evaluation_success', False):
                        overall_eval = result.get('overall_evaluation', {})
                        overall_score = overall_eval.get('overall_average_score', 0)
                        if overall_score > 0:
                            quality_scores.append(overall_score)
                
                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    print(f"Average quality score: {avg_quality:.2f}/10")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final evaluation report"""
        total_time = time.time() - self.start_time
        
        final_report = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'config': self.config.to_dict(),
                'layers_executed': list(self.results.keys())
            },
            'layer_results': self.results,
            'overall_summary': self._calculate_overall_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save comprehensive report
        report_file = os.path.join(
            self.config.results_dir, 
            f'comprehensive_evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        print("=" * 60)
        print("EVALUATION PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Comprehensive report saved to: {report_file}")
        print()
        
        # Print overall summary
        overall_summary = final_report['overall_summary']
        print("OVERALL SUMMARY:")
        print(f"Total test cases processed: {overall_summary.get('total_test_cases', 0)}")
        print(f"Workflows generated: {overall_summary.get('workflows_generated', 0)}")
        print(f"Workflows executed: {overall_summary.get('workflows_executed', 0)}")
        print(f"Output evaluations completed: {overall_summary.get('output_evaluations', 0)}")
        
        if 'average_quality_score' in overall_summary:
            print(f"Average output quality score: {overall_summary['average_quality_score']:.2f}/10")
        
        print()
        
        return final_report
    
    def _calculate_overall_summary(self) -> Dict[str, Any]:
        """Calculate overall pipeline statistics"""
        summary = {
            'total_test_cases': 0,
            'workflows_generated': 0,
            'workflows_executed': 0,
            'output_evaluations': 0
        }
        
        # Layer 1 statistics
        if 'layer_1' in self.results:
            layer_1 = self.results['layer_1']
            summary['total_test_cases'] = layer_1.get('summary', {}).get('total_tests', 0)
            summary['workflows_generated'] = layer_1.get('summary', {}).get('successful', 0)
        
        # Layer 2 statistics
        if 'layer_2' in self.results:
            layer_2 = self.results['layer_2']
            summary['workflows_executed'] = layer_2.get('summary', {}).get('successful', 0)
        
        # Layer 3 statistics
        if 'layer_3' in self.results:
            layer_3 = self.results['layer_3']
            summary['output_evaluations'] = layer_3.get('summary', {}).get('successful', 0)
            
            # Calculate average quality score
            test_results = layer_3.get('test_results', [])
            quality_scores = []
            for result in test_results:
                if result.get('evaluation_success', False):
                    overall_eval = result.get('overall_evaluation', {})
                    overall_score = overall_eval.get('overall_average_score', 0)
                    if overall_score > 0:
                        quality_scores.append(overall_score)
            
            if quality_scores:
                summary['average_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Analyze results and generate suggestions
        if 'layer_1' in self.results:
            layer_1_success_rate = (
                self.results['layer_1'].get('summary', {}).get('successful', 0) /
                max(self.results['layer_1'].get('summary', {}).get('total_tests', 1), 1)
            )
            
            if layer_1_success_rate < 0.8:
                recommendations.append(
                    "Low workflow generation success rate. Consider improving task planning logic."
                )
        
        if 'layer_2' in self.results:
            layer_2_results = self.results['layer_2'].get('test_results', [])
            execution_failures = [r for r in layer_2_results if not r.get('overall_success', False)]
            
            if len(execution_failures) > len(layer_2_results) * 0.3:
                recommendations.append(
                    "High execution failure rate. Review workflow execution logic and error handling."
                )
        
        if 'layer_3' in self.results:
            layer_3_summary = self.results['layer_3'].get('overall_summary', {})
            avg_quality = layer_3_summary.get('average_quality_score', 0)
            
            if avg_quality < 6.0:
                recommendations.append(
                    "Low average output quality. Consider improving agent prompts and output validation."
                )
        
        if not recommendations:
            recommendations.append("Evaluation results look good! Continue monitoring performance.")
        
        return recommendations


def main():
    """Main entry point for evaluation script"""
    parser = argparse.ArgumentParser(description='Workflow Evaluation Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--layers', type=str, default='1,2,3', 
                       help='Comma-separated list of layers to run (e.g., "1,2" or "3")')
    parser.add_argument('--max-processes', type=int, 
                       help='Maximum number of processes to use')
    parser.add_argument('--batch-size', type=int, 
                       help='Batch size for parallel processing')
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing test data')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config and os.path.exists(args.config):
        # Load from config file if provided
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = EvaluationConfig(**config_dict)
    else:
        # Use default configuration
        config = EvaluationConfig()
    
    # Override with command line arguments
    if args.layers:
        config.layers_to_run = [int(x.strip()) for x in args.layers.split(',')]
    
    if args.max_processes:
        config.max_processes = args.max_processes
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.data_dir:
        config.eval_data_dir = args.data_dir
    
    # Validate configuration
    if config.max_processes <= 0:
        config.max_processes = get_optimal_process_count()
    
    try:
        # Run evaluation pipeline
        pipeline = WorkflowEvaluationPipeline(config)
        results = pipeline.run_complete_evaluation()
        
        print("Evaluation pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Evaluation pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
