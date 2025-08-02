"""
User Query Router Component

This component is responsible for processing user queries related to AI Agent workflow modifications.
It classifies user intents, handles composite queries, and routes them to appropriate downstream processors.

The component supports the following query types:
- Addition: Add new elements (nodes, agents, functionality)
- Deletion: Remove existing elements (nodes, agents, functionality)  
- Modification: Update existing elements or their properties
- Restructure: Structural adjustments involving multiple elements
- Query: Information requests about current workflow
- Optimization: Performance improvement suggestions
"""
from evoagentx.core import BaseModule
from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLM, OpenAILLMConfig, BaseLLM
from evoagentx.models.model_configs import LLMConfig
from evoagentx.app.components import QueryType
from evoagentx.prompts.app_prompts import (
    QueryRewritingPromptTemplate, UnifiedClassificationDecompositionPromptTemplate, QueryClarityCheckPromptTemplate
)

from typing import List, Dict, Any, Optional, Union
import json
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from pydantic import BaseModel, Field

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ParsedIntent(BaseModel):
    """Represents a single parsed intent from user query."""
    
    intent_type: QueryType
    target_object: Optional[str] = None  # ID or name of target node/agent
    description: str = ""  # Detailed description of the intent
    parameters: Dict[str, Any] = Field(default_factory=dict)  # Additional parameters for the operation
    confidence: float = 0.0  # Confidence score from LLM parsing
    clear_enough: bool = False  # Whether the intent is clear enough


class QueryParseResult(BaseModel):
    """Result of query parsing containing all identified intents."""
    
    original_query: str
    classified_operations: List[Dict[str, Any]] = Field(default_factory=list)  # Unified format: [{"category": "...", "atomic_query": "...", "not_clear": bool}]
    is_composite: bool = False  # Whether query contains multiple intents
    workflow_context: Dict[str, Any] = Field(default_factory=dict)  # Current workflow state for context
    total_operations: int = 0  # Total number of operations found
    has_frontend: bool = False  # Whether frontend operations are present
    has_backend: bool = False  # Whether backend operations are present


class UserQueryRouter(BaseModule):
    """
    Main component for routing and processing user queries related to workflow modifications.
    
    This component uses LLM-based intent recognition to classify user requests and
    prepare them for downstream processing. It handles both simple and composite
    queries, performing necessary query rewriting and intent extraction.
    """
    
    llm_model_name: str = Field(default="gpt-4o-mini")
    max_retries: int = Field(default=3)
    temperature: float = Field(default=0.1)
    enable_traditional_ml: bool = Field(default=False)
    logger: Any = Field(default=logger)

    def __init__(self):
        """
        Initialize the UserQueryRouter with configuration parameters.
        
        Args:
            llm_model_name: Name of the LLM model to use for intent recognition
            max_retries: Maximum number of retries for LLM API calls
            temperature: Temperature setting for LLM to control randomness
            enable_traditional_ml: Whether to enable traditional ML fallback methods
            logger: Logger instance for debugging and monitoring
        """
        # First initialize the BaseModule/Pydantic model
        super().__init__()
        
        # Initialize LLM client and other components
        self._llm_client = None
        self._traditional_ml_classifier = None
        
        # Prompt templates for different scenarios
        self._query_rewrite_prompt = ""
        self._unified_classification_decomposition_prompt = ""
        self._query_clarity_check_prompt = ""
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize LLM client and other necessary components."""
        # Initialize LLM client based on model name
        self._initialize_llm_client()
        
        # Load traditional ML models if enabled
        if self.enable_traditional_ml:
            self._initialize_traditional_ml()
        
        # Load and prepare prompt templates
        self._initialize_prompt_templates()
    
    def _initialize_llm_client(self) -> None:
        """Initialize the LLM client based on the specified model name."""
        try:
            # Check if we have API key available
            if OPENAI_API_KEY and OPENAI_API_KEY.strip():
                llm_config = OpenAILLMConfig(
                    model=self.llm_model_name,
                    openai_key=OPENAI_API_KEY,
                    temperature=self.temperature,
                    output_response=True
                )
                self._llm_client = OpenAILLM(llm_config)
                self.logger.info(f"Initialized OpenAI LLM client with model: {self.llm_model_name}")
            else:
                # Use mock LLM for testing or when API key is not available
                from evoagentx.hitl.workflow_editor import MockLLM, MockLLMConfig
                mock_config = MockLLMConfig(
                    llm_type="MockLLM",
                    model="mock-model",
                    output_response=True
                )
                self._llm_client = MockLLM(mock_config)
                self.logger.warning("Using MockLLM due to missing OpenAI API key")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {str(e)}")
            # Fallback to mock LLM
            from evoagentx.hitl.workflow_editor import MockLLM, MockLLMConfig
            mock_config = MockLLMConfig(
                llm_type="MockLLM", 
                model="mock-model",
                output_response=True
            )
            self._llm_client = MockLLM(mock_config)
    
    def _initialize_traditional_ml(self) -> None:
        """Initialize traditional ML models for intent classification."""
        # TODO: Implement traditional ML model loading
        self.logger.info("Traditional ML models not yet implemented")
        pass
    
    def _initialize_prompt_templates(self) -> None:
        """Initialize prompt templates for different classification scenarios."""
        # Query rewrite prompt
        self._query_rewrite_prompt = QueryRewritingPromptTemplate().render_complete_prompt_template()
        # unified classification and decomposition prompt
        self._unified_classification_decomposition_prompt = UnifiedClassificationDecompositionPromptTemplate().render_complete_prompt_template()
        # Query clarity check prompt (without workflow context by default)
        self._query_clarity_check_prompt = QueryClarityCheckPromptTemplate(use_workflow_context=False).render_complete_prompt_template()

        self.logger.info("Prompt templates initialized successfully")
    
    def route_query(
        self,
        user_query: str,
        workflow_context: Dict[str, Any],
        rewrite_query: bool = False
    ) -> QueryParseResult:
        """
        Main entry point for processing user queries.
        
        This method uses a simplified approach:
        1. Unified classification and decomposition in a single LLM call
        2. Optional query rewriting for each atomic operation
        3. Clarity check for each atomic operation
        
        Args:
            user_query: Raw user input describing desired modifications(both front end and workflow side are available)
            workflow_context: Current workflow state including nodes, agents, and structure
            rewrite_query: Whether to perform query rewriting on atomic operations
            
        Returns:
            QueryParseResult containing classified operations with clarity indicators
        """
        self.logger.info(f"Processing user query: {user_query[:100]}...")
        
        try:
            # Step 1: Unified classification and decomposition
            self.logger.debug("Step 1: Performing unified classification and decomposition")
            classified_operations = self.unified_classify_and_decompose(user_query, workflow_context)
            
            # Step 2: Process each atomic operation
            processed_operations = []
            for i, operation in enumerate(classified_operations):
                try:
                    category = operation["category"]
                    atomic_query = operation["atomic_query"]
                    
                    # Step 2a: Optional query rewriting
                    final_query = atomic_query
                    if rewrite_query:
                        self.logger.debug(f"Step 2a: Rewriting atomic query {i+1}")
                        try:
                            query_type = QueryType(category)
                            rewritten = self.rewrite_query(atomic_query, query_type, workflow_context)
                            if rewritten and rewritten.strip() != atomic_query.strip():
                                final_query = rewritten
                                self.logger.debug(f"Rewrote query {i+1}: '{atomic_query}' -> '{final_query}'")
                        except Exception as e:
                            self.logger.warning(f"Failed to rewrite query {i+1}: {str(e)}")
                    
                    # Step 2b: Check if the atomic query is clear enough
                    clarity_result = self._check_query_clarity(final_query, category, workflow_context, max_questions=3)
                    
                    processed_operations.append({
                        "category": category,
                        "atomic_query": final_query,
                        "not_clear": not clarity_result["is_clear"],
                        "follow_up_questions": clarity_result.get("follow_up_questions", []),
                        "clarity_reasoning": clarity_result.get("reasoning", "")
                    })
                    
                    self.logger.debug(f"Processed operation {i+1}: {category} -> {final_query} (clear: {clarity_result['is_clear']})")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing operation {i+1}: {str(e)}")
                    # Keep the original operation but mark as unclear
                    processed_operations.append({
                        "category": operation.get("category", "query"),
                        "atomic_query": operation.get("atomic_query", "Unknown operation"),
                        "not_clear": True,
                        "follow_up_questions": ["Could you provide more details about this operation?"],
                        "clarity_reasoning": f"Error processing operation: {str(e)}"
                    })
            
            # Initialize result object
            result = QueryParseResult(
                original_query=user_query,
                classified_operations=processed_operations,
                workflow_context=workflow_context,
                total_operations=len(processed_operations),
                has_frontend=any(op["category"] == "front_end" for op in processed_operations),
                has_backend=any(op["category"] != "front_end" for op in processed_operations),
                is_composite=len(processed_operations) > 1
            )
            
            # Log final processing summary
            unclear_count = sum(1 for op in processed_operations if op["not_clear"])
            self.logger.info(f"Query processing completed. Found {len(processed_operations)} operations: "
                           f"{result.total_operations} total, frontend: {result.has_frontend}, "
                           f"backend: {result.has_backend}, composite: {result.is_composite}, "
                           f"unclear: {unclear_count}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Critical error in route_query: {str(e)}")
            # Return a fallback result with error information
            fallback_result = QueryParseResult(
                original_query=user_query,
                classified_operations=[{
                    "category": "query",
                    "atomic_query": f"Error processing query: {str(e)}",
                    "not_clear": True,
                    "follow_up_questions": ["Could you please rephrase your request?"],
                    "clarity_reasoning": "Critical processing error occurred"
                }],
                workflow_context=workflow_context,
                is_composite=False,
                total_operations=1,
                has_frontend=False,
                has_backend=False
            )
            return fallback_result
        
    def _check_query_clarity(
        self,
        atomic_query: str,
        category: str,
        workflow_context: Dict[str, Any],
        max_questions: int = 3
    ) -> Dict[str, Any]:
        """
        Check if an atomic query is clear and specific enough for execution using LLM.
        
        This method uses LLM to evaluate whether the atomic query contains enough details
        to be actionable, or if it requires further clarification from the user.
        
        Args:
            atomic_query: The atomic query to check
            category: The query category (e.g., "add_node", "front_end")
            workflow_context: Current workflow state for context
            max_questions: Maximum number of follow-up questions to generate
            
        Returns:
            Dictionary containing:
            - "is_clear": bool - Whether the query is clear enough
            - "reasoning": str - Explanation of the clarity assessment
            - "follow_up_questions": List[str] - Questions to clarify unclear queries
        """
        self.logger.debug(f"Checking clarity for query: {atomic_query[:100]}... (category: {category})")
        
        try:
            # Prepare workflow context summary for the prompt
            context_summary = self._prepare_workflow_context_summary(workflow_context)
            
            # Create the clarity check prompt template with workflow context if available
            use_workflow_context = context_summary is not None
            clarity_template = QueryClarityCheckPromptTemplate(use_workflow_context=use_workflow_context)
            
            # Get the base prompt template
            base_prompt = clarity_template.render_complete_prompt_template()
            
            # Format the prompt with actual query, category, context and max_questions
            if use_workflow_context:
                formatted_prompt = base_prompt.format(
                    category=category,
                    atomic_query=atomic_query,
                    workflow_context=context_summary,
                    max_questions=max_questions
                )
            else:
                formatted_prompt = base_prompt.format(
                    category=category,
                    atomic_query=atomic_query,
                    max_questions=max_questions
                )
            
            self.logger.debug("Calling LLM for query clarity check")
            
            # Call LLM for clarity assessment (tenacity will handle retries automatically)
            response = self._call_llm(formatted_prompt, expected_format="json")
            
            # Parse the LLM response
            is_clear = response.get("is_clear", False)
            reasoning = response.get("reasoning", "No reasoning provided")
            follow_up_questions = response.get("follow_up_questions", [])
            
            # Validate and clean up follow-up questions
            if isinstance(follow_up_questions, list):
                # Limit the number of questions and ensure they are strings
                follow_up_questions = [
                    str(q).strip() for q in follow_up_questions[:max_questions] 
                    if q and str(q).strip()
                ]
            else:
                follow_up_questions = []
            
            # If marked as unclear but no follow-up questions provided, add a generic one
            if not is_clear and not follow_up_questions:
                follow_up_questions = [f"Could you provide more specific details about this {category} operation?"]
            
            result = {
                "is_clear": is_clear,
                "reasoning": reasoning,
                "follow_up_questions": follow_up_questions
            }
            
            self.logger.debug(f"Clarity check result: clear={is_clear}, questions={len(follow_up_questions)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in LLM clarity check: {str(e)}")
            # Fallback to a conservative approach - mark as unclear and ask for clarification
            return {
                "is_clear": False,
                "reasoning": f"Could not assess clarity due to error: {str(e)}",
                "follow_up_questions": [
                    f"Could you provide more specific details about this {category} operation?",
                    "What exactly would you like to accomplish with this request?"
                ][:max_questions]
            }
    
    def unified_classify_and_decompose(
        self,
        query: str,
        workflow_context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Unified method to classify and decompose user queries into atomic operations.
        
        This method combines the functionality of classify_intent and decompose_composite_query
        into a single LLM call for better efficiency and consistency.
        
        Args:
            query: Raw user query to process
            workflow_context: Current workflow state for context
            
        Returns:
            List of dictionaries, each containing:
            - "category": The query type category (e.g., "add_node", "front_end")
            - "atomic_query": The specific atomic operation description
        """
        self.logger.debug(f"Performing unified classification and decomposition for query: {query[:100]}...")
        
        try:
            # Prepare workflow context summary for the prompt
            context_summary = self._prepare_workflow_context_summary(workflow_context)
            
            # Create the unified prompt template with workflow context if available
            use_workflow_context = context_summary is not None
            unified_template = UnifiedClassificationDecompositionPromptTemplate(use_workflow_context=use_workflow_context)
            
            # Get the base prompt template
            base_prompt = unified_template.render_complete_prompt_template()
            
            # Format the prompt with actual query and context
            if use_workflow_context:
                formatted_prompt = base_prompt.format(
                    user_query=query,
                    workflow_context=context_summary
                )
            else:
                formatted_prompt = base_prompt.format(user_query=query)
            
            self.logger.debug("Calling LLM for unified classification and decomposition")
            
            # Call LLM for unified processing (tenacity will handle retries automatically)
            response = self._call_llm(formatted_prompt, expected_format="json")
            
            # Parse the LLM response
            classified_operations = response.get("classified_operations", [])
            total_operations = response.get("total_operations", len(classified_operations))
            has_frontend = response.get("has_frontend", False)
            has_backend = response.get("has_backend", False)
            
            self.logger.debug(f"Unified processing result: {total_operations} operations, frontend: {has_frontend}, backend: {has_backend}")
            
            # Validate and clean up the classified operations
            validated_operations = []
            for i, operation in enumerate(classified_operations):
                if isinstance(operation, dict) and "category" in operation and "atomic_query" in operation:
                    category = operation["category"].strip()
                    atomic_query = operation["atomic_query"].strip()
                    
                    # Validate category against known QueryType values
                    try:
                        # Check if it's a valid QueryType
                        QueryType(category)
                        validated_operations.append({
                            "category": category,
                            "atomic_query": atomic_query
                        })
                    except ValueError:
                        self.logger.warning(f"Unknown category '{category}' in operation {i+1}, skipping")
                else:
                    self.logger.warning(f"Invalid operation format at index {i}: {operation}")
            
            # Log successful processing
            self.logger.info(f"Successfully processed query into {len(validated_operations)} classified operations")
            for i, operation in enumerate(validated_operations):
                self.logger.debug(f"Operation {i+1}: {operation['category']} -> {operation['atomic_query']}")
            
            return validated_operations
            
        except Exception as e:
            self.logger.error(f"Error in unified classification and decomposition: {str(e)}")
            # Re-raise the exception to be handled by calling code
            raise RuntimeError(f"Unified query processing failed: {str(e)}") from e
    
    def classify_intent(
        self,
        query: str,
        workflow_context: Dict[str, Any]
    ) -> List[QueryType]:
        """
        Classify the intent type(s) present in the user query.
        
        Uses LLM-based classification to identify which types of operations
        the user is requesting. Handles both single and multiple intents.
        
        Args:
            query: User query to classify
            workflow_context: Current workflow state for context, this is optional
            
        Returns:
            List of identified query types
        
        Note: This method is now deprecated. Use unified_classify_and_decompose instead.
        """
        self.logger.warning("classify_intent method is deprecated. Use unified_classify_and_decompose instead.")
        return []
    
    def _prepare_workflow_context_summary(self, workflow_context: Dict[str, Any]) -> str:
        """
        Prepare a concise summary of workflow context for the LLM prompt.
        
        Args:
            workflow_context: Current workflow state
            
        Returns:
            Formatted context summary string
        """
        # # not used for now
        # return None
        if not workflow_context:
            return None
        
        try:
            # Extract key information from workflow context
            summary_parts = []
            
            # Add basic workflow info
            if "workflow_id" in workflow_context:
                summary_parts.append(f"Workflow ID: {workflow_context.get('workflow_id')}")
            
            # Add node information
            if "graph" in workflow_context and isinstance(workflow_context["graph"], dict):
                graph = workflow_context["graph"]
                if "nodes" in graph:
                    nodes = graph["nodes"]
                    if isinstance(nodes, list):
                        summary_parts.append(f"Number of nodes: {len(nodes)}")
                        # Add node names if available
                        node_names = [node.get("name", f"Node-{i}") for i, node in enumerate(nodes[:5])]
                        if node_names:
                            summary_parts.append(f"Node names: {', '.join(node_names)}")
                            if len(nodes) > 5:
                                summary_parts.append(f"... and {len(nodes) - 5} more nodes")
            
            # Add agent information if available
            if "agents" in workflow_context:
                agents = workflow_context["agents"]
                if isinstance(agents, list):
                    summary_parts.append(f"Number of agents: {len(agents)}")
            
            return "\n".join(summary_parts) if summary_parts else "Basic workflow structure available."
            
        except Exception as e:
            self.logger.warning(f"Error preparing workflow context summary: {str(e)}")
            return "Workflow context available but could not be summarized."
    
    def rewrite_query(
        self,
        query: str,
        intent_type: QueryType,
        workflow_context: Dict[str, Any]
    ) -> str:
        """
        Rewrite user query for better clarity and processing.
        
        Sometimes user queries are ambiguous or lack specific details.
        This method generates clearer, more actionable versions of the query.
        
        Args:
            query: Original user query
            intent_type: Identified intent type
            workflow_context: Current workflow state for context
            
        Returns:
            Rewritten query with improved clarity
        """
        # TODO: Implement query rewriting logic
        pass
    
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def _call_llm(
        self,
        prompt: str,
        expected_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Make a call to the LLM with error handling and retries using tenacity.
        
        The retry mechanism will automatically retry on failures with exponential backoff:
        - Maximum 3 attempts
        - Wait time: 1s, 2s, 4s (exponential backoff)
        - Maximum wait time: 10s
        
        Args:
            prompt: Formatted prompt for the LLM
            expected_format: Expected response format (json, text, etc.)
            
        Returns:
            Parsed LLM response
            
        Raises:
            RuntimeError: If LLM client is not initialized
            ValueError: If JSON parsing fails consistently
            Exception: Any other LLM-related errors after all retries
        """
        if self._llm_client is None:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Formulate messages for the LLM
            messages = self._llm_client.formulate_messages([prompt])
            
            # Generate response
            response = self._llm_client.single_generate(messages[0])
            
            self.logger.debug(f"LLM response: {response[:200]}...")
            
            # Parse response based on expected format
            if expected_format == "json":
                # Try to parse JSON from the response
                try:
                    # Clean up the response to extract JSON
                    response_clean = response.strip()
                    if response_clean.startswith("```json"):
                        response_clean = response_clean[7:]
                    if response_clean.endswith("```"):
                        response_clean = response_clean[:-3]
                    
                    parsed_response = json.loads(response_clean)
                    return parsed_response
                    
                except json.JSONDecodeError:
                    # If direct JSON parsing fails, try to extract JSON from text
                    from evoagentx.core.module_utils import parse_json_from_text
                    json_data = parse_json_from_text(response)
                    if json_data:
                        return json_data
                    else:
                        # This will trigger a retry
                        raise ValueError(f"Failed to parse JSON from response: {response[:500]}...")
                        
            elif expected_format == "text":
                return {"text": response}
            else:
                return {"response": response}
                
        except Exception as e:
            # Log the attempt failure - tenacity will handle the retry
            self.logger.warning(f"LLM call failed: {str(e)}")
            
            # For some specific errors, we might want to provide fallback info
            if self.enable_traditional_ml and "JSON" not in str(e):
                # Only suggest fallback for non-JSON parsing errors
                self.logger.debug("Traditional ML fallback available if all retries fail")
            
            # Re-raise to let tenacity handle the retry
            raise
    
    def _traditional_ml_strategy(
        self,
        query: str
    ) -> QueryType:
        """
        Used when hybrid mode or pure ML mode is enabled.
        
        Args:
            query: User query to classify
            
        Returns:
            Classified query type using traditional ML methods
        """
        # TODO: Implement traditional ML fallback classification
        pass
    
    def get_supported_query_types(self) -> List[str]:
        """
        Get list of all supported query types.
        
        Returns:
            List of supported query type names
        """
        return [query_type.value for query_type in QueryType]
    
    def get_component_status(self) -> Dict[str, Any]:
        """
        Get current status and configuration of the router component.
        
        Returns:
            Dictionary containing component status and configuration
        """
        return {
            "llm_model": self.llm_model_name,
            "traditional_ml_enabled": self.enable_traditional_ml,
            "supported_types": self.get_supported_query_types(),
            "component_ready": self._llm_client is not None
        }
