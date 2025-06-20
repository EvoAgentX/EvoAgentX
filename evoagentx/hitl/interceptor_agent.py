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
    pass

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


class HITLConversationAgent(HITLBaseAgent):
    pass

class HITLConversationAction(Action):
    pass