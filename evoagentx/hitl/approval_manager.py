import asyncio
import json
from typing import Dict
from pydantic import Field
from ..core.logging import logger
from ..core import BaseModule
from .hitl import HITLRequest, HITLResponse, HITLDecision, HITLContext, HITLInteractionType, HITLMode

class HITLManager(BaseModule):
    """
    HITL Manager - Manages Human-in-the-Loop interactions
    It must be instancialized and add as a parameter to the WorkFlow instance like: workflow = WorkFlow(graph=graph, llm=llm, agent_manager=AgentManager(agents=agents), hitl_manager=hitl_manager)
    """
    
    # Pydantic fields
    active: bool = Field(default=False, description="Whether HITL is currently active")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self.hitl_input_output_mapping = {}
    
    def init_module(self):
        """Module initialization"""
        self._pending_requests: Dict[str, asyncio.Future] = {}
    
    def activate(self):
        """activate HITL feature"""
        self.active = True
        logger.info("HITL feature activated")
    
    def deactivate(self):
        """deactivate HITL feature"""
        self.active = False
        logger.info("HITL feature deactivated")
    
    @property
    def is_active(self) -> bool:
        return self.active

    async def request_approval(
        self,
        task_name: str,
        agent_name: str, 
        action_name: str,
        interaction_type: HITLInteractionType,
        mode: HITLMode,
        action_inputs_data: dict = None,
        execution_result = None,
        workflow_goal: str = None,
        display_context: Dict = None,
        timeout: float = 1800.0
    ) -> HITLResponse:
        """Request human approval"""
        
        if not self.active:
            # HITL is not active, auto-approved
            return HITLResponse(
                request_id="auto_approved",
                decision=HITLDecision.APPROVE,
                feedback="HITL not active, auto-approved"
            )
        
        # create HITL context
        context = HITLContext(
            task_name=task_name,
            agent_name=agent_name,
            action_name=action_name,
            workflow_goal=workflow_goal,
            action_inputs=action_inputs_data or {},
            execution_result=execution_result,
            display_context=display_context or {}
        )
        
        # generate prompt message
        prompt_message = self._generate_prompt_message(interaction_type, mode, context)
        
        # create request
        request = HITLRequest(
            interaction_type=interaction_type,
            mode=mode,
            context=context,
            prompt_message=prompt_message
        )
        
        # create Future to wait for response
        future = asyncio.Future()
        self._pending_requests[request.request_id] = future
        
        # display request and wait for response in CLI
        try:
            response = await self._handle_cli_interaction(request, timeout)
            future.set_result(response)
            return response
        except asyncio.TimeoutError:
            response = HITLResponse(
                request_id=request.request_id,
                decision=HITLDecision.REJECT,
                feedback="Timeout: No human response received"
            )
            future.set_result(response)
            return response
        finally:
            # clean up
            self._pending_requests.pop(request.request_id, None)
    
    async def _handle_cli_interaction(self, request: HITLRequest, timeout: float) -> HITLResponse:
        """handle cli interaction"""
        
        print("\n" + "="*80)
        print("🔔 Human-in-the-Loop approval request")
        print("="*80)
        print(request.prompt_message)
        print("="*80)
        
        try:
            if request.interaction_type == HITLInteractionType.APPROVE_REJECT:
                return await self._handle_approve_reject(request)
            elif request.interaction_type == HITLInteractionType.REVIEW_EDIT_STATE:
                return await self._handle_review_edit(request)
            elif request.interaction_type == HITLInteractionType.REVIEW_TOOL_CALLS:
                return await self._handle_tool_calls(request)
            elif request.interaction_type == HITLInteractionType.MULTI_TURN_CONVERSATION:
                return await self._handle_conversation(request)
            else:
                return HITLResponse(
                    request_id=request.request_id,
                    decision=HITLDecision.REJECT,
                    feedback="Unknown interaction type"
                )
        except Exception as e:
            logger.error(f"CLI interaction error: {e}")
            return HITLResponse(
                request_id=request.request_id,
                decision=HITLDecision.REJECT,
                feedback=f"Error: {str(e)}"
            )
    
    async def _handle_approve_reject(self, request: HITLRequest) -> HITLResponse:
        """handle approve/reject"""
        
        def get_user_input():
            while True:
                choice = input("\nPlease select [a]pprove / [r]eject: ").lower().strip()
                if choice in ['a', 'approve']:
                    return HITLDecision.APPROVE
                elif choice in ['r', 'reject']:
                    return HITLDecision.REJECT
                print("Invalid input, please input 'a' or 'r'")
        
        # run blocking input in event loop
        loop = asyncio.get_event_loop()
        decision = await loop.run_in_executor(None, get_user_input)
        
        feedback = ""
        if decision == HITLDecision.REJECT:
            def get_feedback():
                return input("Please provide the reason for rejection (optional): ").strip()
            feedback = await loop.run_in_executor(None, get_feedback)
        
        return HITLResponse(
            request_id=request.request_id,
            decision=decision,
            feedback=feedback if feedback else None
        )
    
    async def _handle_review_edit(self, request: HITLRequest) -> HITLResponse:
        """handle review edit"""
        # TODO: implement review edit
        raise NotImplementedError("Not implemented HITL type: HITLInteractionType.REVIEW_EDIT_STATE")
        # def get_user_input():
        #     print(f"\nCurrent execution result:")
        #     result = request.context.execution_result
        #     if isinstance(result, dict):
        #         print(json.dumps(result, ensure_ascii=False, indent=2))
        #     else:
        #         print(str(result))
            
        #     while True:
        #         choice = input("\nPlease select [a]pprove / [m]odify / [r]eject: ").lower().strip()
        #         if choice in ['a', 'approve']:
        #             return HITLDecision.APPROVE, None
        #         elif choice in ['r', 'reject']:
        #             return HITLDecision.REJECT, None
        #         elif choice in ['m', 'modify']:
        #             new_content = input("Please input the modified content (JSON format): ").strip()
        #             try:
        #                 if new_content:
        #                     modified = json.loads(new_content)
        #                     return HITLDecision.MODIFY, modified
        #                 else:
        #                     return HITLDecision.APPROVE, None
        #             except json.JSONDecodeError:
        #                 print("JSON format error, please input again")
        #                 continue
        #         print("Invalid input, please input 'a', 'm' or 'r'")
        
        # loop = asyncio.get_event_loop()
        # decision, modified_content = await loop.run_in_executor(None, get_user_input)
        
        # return HITLResponse(
        #     request_id=request.request_id,
        #     decision=decision,
        #     modified_content=modified_content
        # )
    
    async def _handle_tool_calls(self, request: HITLRequest) -> HITLResponse:
        """handle tool calls review"""
        # TODO: implement tool calls review
        raise NotImplementedError("Not implemented HITL type: HITLInteractionType.REVIEW_TOOL_CALLS")
        # def get_user_input():
        #     tool_calls = request.context.action_inputs.get('tool_calls', [])
        #     print(f"\nTool calls:")
        #     print(json.dumps(tool_calls, ensure_ascii=False, indent=2))
            
        #     while True:
        #         choice = input("\nPlease select [a]pprove / [m]odify / [r]eject: ").lower().strip()
        #         if choice in ['a', 'approve']:
        #             return HITLDecision.APPROVE, None
        #         elif choice in ['r', 'reject']:
        #             return HITLDecision.REJECT, None
        #         elif choice in ['m', 'modify']:
        #             new_calls = input("Please input the modified tool calls (JSON format): ").strip()
        #             try:
        #                 if new_calls:
        #                     modified = json.loads(new_calls)
        #                     return HITLDecision.MODIFY, modified
        #                 else:
        #                     return HITLDecision.APPROVE, None
        #             except json.JSONDecodeError:
        #                 print("JSON format
        
        # loop = asyncio.get_event_loop()
        # decision, modified_content = await loop.run_in_executor(None, get_user_input)
        
        # return HITLResponse(
        #     request_id=request.request_id,
        #     decision=decision,
        #     modified_content=modified_content
        # )
    
    async def _handle_conversation(self, request: HITLRequest) -> HITLResponse:
        """handle multi-turn conversation"""
        # TODO: implement multi-turn conversation
        raise NotImplementedError("Not implemented HITL type: HITLInteractionType.MULTI_TURN_CONVERSATION")
        # def get_user_input():
        #     user_input = input("\nPlease input the guidance content (or 'continue' to continue): ").strip()
        #     if user_input.lower() == 'continue':
        #         return HITLDecision.CONTINUE, None
        #     elif user_input.lower() in ['reject', 'stop']:
        #         return HITLDecision.REJECT, None
        #     else:
        #         return HITLDecision.MODIFY, user_input
        
        # loop = asyncio.get_event_loop()
        # decision, content = await loop.run_in_executor(None, get_user_input)
        
        # return HITLResponse(
        #     request_id=request.request_id,
        #     decision=decision,
        #     modified_content=content,
        #     feedback=content
        # )
    
    def _generate_prompt_message(
        self, 
        interaction_type: HITLInteractionType, 
        mode: HITLMode, 
        context: HITLContext
    ) -> str:
        """generate prompt message"""
        
        base_info = f"""
Task: {context.task_name}
Agent: {context.agent_name}
Action: {context.action_name}
Workflow Goal: {context.workflow_goal or 'N/A'}
Mode: {'Pre-Execution Approval' if mode == HITLMode.PRE_EXECUTION else 'Post-Execution Review'}
"""
        
        if mode == HITLMode.PRE_EXECUTION:
            base_info += f"\nparameters to be executed:\n{json.dumps(context.action_inputs, ensure_ascii=False, indent=2)}"
        else:
            base_info += f"\nexecution_result:\n{json.dumps(context.execution_result, ensure_ascii=False, indent=2) if context.execution_result else 'None'}"
        
        return base_info
