# evoagentx/hitl/interceptor_agent.py

import asyncio
import sys
from typing import Tuple
from ..agents.agent import Agent
from ..actions.action import Action
from .approval_manager import HITLManager
from .hitl import HITLInteractionType, HITLMode, HITLDecision
from ..core.registry import MODULE_REGISTRY
from ..core.logging import logger

class HITLInterceptorAction(Action):
    """HITL Interceptor Action"""
    
    def __init__(
        self, 
        target_agent_name: str, 
        target_action_name: str,
        name: str = None,
        description: str = "A pre-defined action to proceed the Human-In-The-Loop",
        interaction_type: HITLInteractionType = HITLInteractionType.APPROVE_REJECT,
        mode: HITLMode = HITLMode.PRE_EXECUTION,
        **kwargs
    ):
        if not name:
            name = f"hitl_intercept_{target_agent_name}_{target_action_name}_mode_{mode.value}_action"
        super().__init__(
            name=name,
            description=description,
            **kwargs
        )
        self.target_agent_name = target_agent_name
        self.target_action_name = target_action_name
        self.interaction_type = interaction_type
        self.mode = mode
        
    def execute(self, llm, inputs: dict, hitl_manager: HITLManager, sys_msg: str = None, **kwargs) -> Tuple[dict, str]:
        try:
            # get current running loop
            loop = asyncio.get_running_loop()
            if loop:
                pass
            # if in async context, cannot use asyncio.run()
            raise RuntimeError("Cannot use asyncio.run() in async context. Use async_execute directly.")
        except RuntimeError:
            # if not in async context, use asyncio.run()
            return asyncio.run(self.async_execute(llm, inputs, hitl_manager, sys_msg=sys_msg, **kwargs))
    
    async def async_execute(self, llm, inputs: dict, hitl_manager:HITLManager, sys_msg: str = None, **kwargs) -> Tuple[dict, str]:
        """
        Asynchronous execution of HITL Interceptor
        """
        
        task_name = kwargs.get('wf_task', 'Unknown Task')
        workflow_goal = kwargs.get('wf_goal', None)
        
        # request HITL approval
        response = await hitl_manager.request_approval(
            task_name=task_name,
            agent_name=self.target_agent_name,
            action_name=self.target_action_name,
            interaction_type=self.interaction_type,
            mode=self.mode,
            action_inputs_data=inputs,
            workflow_goal=workflow_goal
        )
        
        result = {
            "hitl_decision": response.decision,
            "target_agent": self.target_agent_name,
            "target_action": self.target_action_name,
            "hitl_feedback": response.feedback
        }
        for output_name in self.outputs_format.get_attrs():
            try:
                result |= {output_name: inputs[hitl_manager.hitl_input_output_mapping[output_name]]}
            except Exception as e:
                logger.exception(e)
        
        prompt = f"HITL Interceptor executed for {self.target_agent_name}.{self.target_action_name}"
        if result["hitl_decision"] == HITLDecision.APPROVE:
            prompt += "\nHITL approved, the action will be executed"
            return result, prompt
        elif result["hitl_decision"] == HITLDecision.REJECT:
            prompt += "\nHITL rejected, the action will not be executed"
            sys.exit()
            # return result, prompt

class HITLPostExecutionAction(Action):
    pass

class HITLBaseAgent(Agent):
    """
    Include all Agent classes for hitl use case
    """
    def _get_unique_class_name(self, candidate_name: str) -> str:
        
        if not MODULE_REGISTRY.has_module(candidate_name):
            return candidate_name 
        
        i = 1 
        while True:
            unique_name = f"{candidate_name}V{i}"
            if not MODULE_REGISTRY.has_module(unique_name):
                break
            i += 1 
        return unique_name

class HITLInterceptorAgent(HITLBaseAgent):
    """HITL Interceptor Agent - Intercept the execution of other agents"""
    
    def __init__(self,
                 target_agent_name: str,
                 target_action_name: str,
                 name: str = None,
                 interaction_type: HITLInteractionType = HITLInteractionType.APPROVE_REJECT,
                 mode: HITLMode = HITLMode.PRE_EXECUTION,
                 **kwargs):
        
        # generate agent name
        if target_action_name:
            agent_name = f"HITL_Interceptor_{target_agent_name}_{target_action_name}_mode_{mode.value}"
        else:
            agent_name = f"HITL_Interceptor_{target_agent_name}_mode_{mode.value}"
        
        super().__init__(
            name=agent_name,
            description=f"HITL Interceptor - Intercept the execution of {target_agent_name}",
            is_human=True,  
            **kwargs
        )
        
        self.target_agent_name = target_agent_name
        self.target_action_name = target_action_name
        self.interaction_type = interaction_type
        self.mode = mode
        
        # add intercept action
        if mode == HITLMode.PRE_EXECUTION:
            action = HITLInterceptorAction(
                target_agent_name=target_agent_name,
                target_action_name=target_action_name or "any",
                interaction_type=interaction_type,
                mode=mode
            )
        elif mode == HITLMode.POST_EXECUTION:
            action = HITLPostExecutionAction(
                target_agent_name=target_agent_name,
                target_action_name=target_action_name or "any",
                interaction_type=interaction_type
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.add_action(action)
        # self.default_action_name = action.name

    def get_hitl_agent_name(self) -> str:
        """
        Get the name of the HITL agent. Useful when the name of HITL agent is generated dynamically.
        """
        return self.name
    

class HITLUserInputCollectorAction(Action):
    """HITL User Input Collector Action - Collect user input for the HITL Interceptor"""
    
    def __init__(
        self,
        name: str = None,
        agent_name: str = None,
        description: str = "A pre-defined action to collect user input for the HITL Interceptor",
        interaction_type: HITLInteractionType = HITLInteractionType.COLLECT_USER_INPUT,
        input_fields: dict = None,
        **kwargs
        ):
        if not name:
            pass # TODO: generate name
        
        super().__init__(name=name, description=description, **kwargs)
        
        self.interaction_type = interaction_type
        self.input_fields = input_fields or {}
        self.agent_name = agent_name

    def execute(self, llm, inputs: dict, hitl_manager: HITLManager, sys_msg: str = None, **kwargs) -> Tuple[dict, str]:
        try:
            # get current running loop
            loop = asyncio.get_running_loop()
            if loop:
                pass
            # if in async context, cannot use asyncio.run()
            raise RuntimeError("Cannot use asyncio.run() in async context. Use async_execute directly.")
        except RuntimeError:
            # if not in async context, use asyncio.run()
            return asyncio.run(self.async_execute(llm, inputs, hitl_manager, sys_msg=sys_msg, **kwargs))

    async def async_execute(self, llm, inputs: dict, hitl_manager: HITLManager, sys_msg: str = None, **kwargs) -> Tuple[dict, str]:
        """
        Asynchronous execution of HITL User Input Collector
        """
    
        task_name = kwargs.get('wf_task', 'Unknown Task')
        workflow_goal = kwargs.get('wf_goal', None)

        # request user input from HITL manager
        response = await hitl_manager.request_user_input(
            task_name=task_name,
            agent_name=self.agent_name,
            action_name=self.name,
            input_fields=self.input_fields,
            workflow_goal=workflow_goal
        )
        
        result = {
            "hitl_decision": response.decision,
            "collected_user_input": response.modified_content or {},
            "hitl_feedback": response.feedback
        }
        
        # Map collected user input to outputs if output format is defined
        if self.outputs_format:
            for output_name in self.outputs_format.get_attrs():
                if output_name in response.modified_content:
                    result[output_name] = response.modified_content[output_name]
        
        prompt = f"HITL User Input Collector executed: {self.name}"
        if result["hitl_decision"] == HITLDecision.CONTINUE:
            prompt += f"\nUser input collection completed: {result['collected_user_input']}"
            return result, prompt
        elif result["hitl_decision"] == HITLDecision.REJECT:
            prompt += "\nUser cancelled input or error occurred"
            sys.exit()

class HITLUserInputCollectorAgent(HITLBaseAgent):
    """HITL User Input Collector Agent - Collect user input for the HITL Interceptor"""
    
    def __init__(self,
                 name: str = None,
                 input_fields: dict = None,
                 interaction_type: HITLInteractionType = HITLInteractionType.COLLECT_USER_INPUT,
                 **kwargs):

        # generate agent name
        if name:
            agent_name = f"HITL_User_Input_Collector_{name}"
        else:
            pass # TODO: generate name

        super().__init__(
            name=agent_name,
            description="HITL User Input Collector - Collect predefined user inputs",
            is_human=True,
            **kwargs
        )

        self.interaction_type = interaction_type
        self.input_fields = input_fields or {}

        # generation Action name
        action_name_validated = False
        name_i = 0
        action_name = None
        while not action_name_validated:
            action_name = "HITLUserInputCollectorAction"+f"_{name_i}"
            if MODULE_REGISTRY.has_module(action_name):
                continue
            else:
                action_name_validated = True
        # add user input collector action
        action = HITLUserInputCollectorAction(
            name=action_name,
            agent_name=agent_name,
            interaction_type=interaction_type,
            input_fields=self.input_fields
        )
        
        self.add_action(action)

    def get_hitl_agent_name(self) -> str:
        """
        Get the name of the HITL agent. Useful when the name of HITL agent is generated dynamically.
        """
        return self.name
    
    def set_input_fields(self, input_fields: dict):
        """Set the input fields for user input collection"""
        self.input_fields = input_fields
        # Update the action's input fields as well
        for action in self.actions:
            if isinstance(action, HITLUserInputCollectorAction):
                action.input_fields = input_fields

class HITLConversationAgent(HITLBaseAgent):
    pass

class HITLConversationAction(Action):
    pass