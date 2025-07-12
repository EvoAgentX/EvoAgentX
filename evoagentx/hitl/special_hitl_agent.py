import json
import asyncio
import sys
from typing import Dict, Any, Tuple
from ..actions.action import Action
from ..models.base_model import BaseLLM
from ..core.logging import logger
from .approval_manager import HITLManager
from .hitl import HITLDecision
from .interceptor_agent import HITLBaseAgent
from .hitl_gui import WorkFlowJSONEditorGUI


class HITLOutsideConversationAction(Action):
    """HITL Outside Conversation Action - support the conversation loop to modify the workflow json structure"""

    def __init__(
        self,
        name: str = "HITLOutsideConversationAction",
        description: str = "support the conversation loop to modify the workflow json structure",
        **kwargs
    ):
        super().__init__(
            name=name,
            description=description,
            **kwargs
        )

    def execute(self, llm: BaseLLM, inputs: dict, hitl_manager: HITLManager, sys_msg: str = None, **kwargs) -> Tuple[dict, str]:
        """synchronous execution entry"""
        try:
            #  check if in the asynchronous context
            loop = asyncio.get_running_loop()
            if loop:
                pass
            # if in the asynchronous context, cannot use asyncio.run()
            raise RuntimeError("Cannot use asyncio.run() in async context. Use async_execute directly.")
        except RuntimeError:
            # if not in the asynchronous context, use asyncio.run()
            return asyncio.run(self.async_execute(llm, inputs, hitl_manager, sys_msg=sys_msg, **kwargs))

    async def async_execute(self, llm: BaseLLM, inputs: dict, hitl_manager: HITLManager, sys_msg: str = None, **kwargs) -> Tuple[dict, str]:
        """
        WorkFlow asynchronously execute the conversation loop to modify the workflow json structure
        Parameters:
            llm: the LLM model
            inputs: the input parameters
            hitl_manager: the HITLManager instance
            sys_msg: the system message
            **kwargs: the additional parameters
        Returns:
            result: the result of the conversation loop, with structure:
                {
                    "final_workflow": the final workflow instance,
                    "workflow_json": the final workflow json structure,
                    "hitl_decision": the HITLDecision of the conversation loop
                }
            prompt: the prompt of the conversation loop
        """
        
        # get the input parameters
        workflow_json_path = inputs.get('workflow_json_path')
        existing_workflow = inputs.get('existing_workflow')
        
        # initialize the workflow json structure
        workflow_json = None
        
        if workflow_json_path:
            # load the workflow json from the json file
            workflow_json = self._load_workflow_info_from_json(workflow_json_path)
        elif existing_workflow:
            # convert the existing workflow to the json structure
            workflow_json = self._convert_workflow_to_json(existing_workflow)
        else:
            raise ValueError("must provide the workflow_json_path or existing_workflow parameter")
            
        # start the conversation loop
        workflow_json = await self._conversation_loop(llm, workflow_json, hitl_manager, **kwargs)
        
        # try to instantiate the workflow
        final_workflow = self._instantiate_workflow(workflow_json, llm)
        
        result = {
            "final_workflow": final_workflow,
            "workflow_json": workflow_json,
            "hitl_decision": HITLDecision.CONTINUE
        }
        
        prompt = "WorkFlow conversation loop finished"
        return result, prompt

    def _load_workflow_info_from_json(self, json_path: str) -> Any:
        """load the workflow info from the json file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"load the workflow info from the json file failed: {e}")
            raise

    def _convert_workflow_to_json(self, workflow) -> Any:
        """convert the workflow to json"""
        try:
            from ..workflow.workflow import WorkFlow
            if not isinstance(workflow, WorkFlow):
                raise TypeError("Expected WorkFlow instance")
            return workflow.graph.to_dict()
        except Exception as e:
            logger.error(f"convert the workflow to json failed: {e}")
            raise


    async def _conversation_loop(self, llm: BaseLLM, workflow_json: Dict[str, Any], hitl_manager: HITLManager, **kwargs) -> Dict[str, Any]:
        """simplified conversation loop - use GUI editor"""
        if not hitl_manager.is_active:
            raise ValueError("HITLManager is not active, please activate the HITLManager first")
        print("\n🎯 WorkFlow JSON editor")
        print("=" * 50)

        original_workflow_json = workflow_json
        
        while True:
            try:
                # try to validate the workflow structure
                workflow_instance = self._instantiate_workflow(workflow_json, llm)
                del workflow_instance
                print("✅ WorkFlow structure validation successful!")
                
                # ask the user how to handle
                print("\nplease choose the operation:")
                print("1. 📝 open the GUI editor(still in development)")
                print("2. 🤖 use the LLM to optimize")
                print("3. 📋 view the current JSON")
                print("4. ✅ finish the edit")
                print("5. ❌ exit")
                print("6. 🔄 reload the original JSON")
                
                choice = input("\nplease choose (1-5): ").strip()
                
                if choice == '1':
                    # open the GUI editor
                    print("🚀 opening the GUI editor...")
                    editor = WorkFlowJSONEditorGUI(workflow_json)
                    edited_json = editor.edit_json()
                    
                    if edited_json is not None:
                        workflow_json = edited_json
                        print("✅ JSON updated")
                    else:
                        print("❌ edit cancelled")
                        
                elif choice == '2':
                    # LLM optimization
                    user_advice = input("please input the optimization advice (type q to cancel): ").strip()
                    if user_advice == "q":
                        continue
                    workflow_json = await self._llm_optimize_workflow(llm, workflow_json, user_advice if user_advice else None)
                    
                elif choice == '3':
                    # view the JSON
                    print("\n📋 current JSON structure:")
                    print(json.dumps(workflow_json, indent=2, ensure_ascii=False))
                    
                elif choice == '4':
                    # finish the edit
                    print("✅ edit finished")
                    break
                    
                elif choice == '5':
                    # exit
                    print("❌ exit the edit")
                    sys.exit()
                    
                elif choice == '6':
                    # reload the original JSON
                    workflow_json = original_workflow_json
                    print("✅ reload the original data")
                    
                else:
                    print("❌ invalid choice, please try again")
                    
            except Exception as e:
                print(f"❌ WorkFlow structure validation failed: {e}")
                print("please fix the JSON structure and try again")
                
                # provide the repair options
                print("\nrepair options:")
                print("1. 📝 open the GUI editor to fix")
                print("2. 🔄 reload the original JSON")
                print("3. ❌ exit")
                
                fix_choice = input("please choose (1-3): ").strip()
                
                if fix_choice == '1':
                    editor = WorkFlowJSONEditorGUI(workflow_json)
                    edited_json = editor.edit_json()
                    if edited_json is not None:
                        workflow_json = edited_json
                elif fix_choice == '2':
                    # reload the original JSON
                    workflow_json = original_workflow_json
                    print("⚠️ reload the original data")
                elif fix_choice == '3':
                    sys.exit()
                    
        return workflow_json

    async def _llm_optimize_workflow(self, llm: BaseLLM, workflow_json: Dict[str, Any], user_advice: str = None) -> Dict[str, Any]:
        """let the LLM optimize the workflow structure"""
        print("🤖 let the LLM optimize the workflow structure...")
        
        optimization_prompt = f"""
        analyze the workflow and optimize it according to the user's advice and make it more reasonable and efficient, make sure to keep the original key of the json dict,and the original structure of the json dict:

        current workflow structure:
        {json.dumps(workflow_json, indent=2, ensure_ascii=False)}

        user's advice:
        {user_advice}

        after the user's advice, please consider the following rules:
        1. the description of the node is clear
        2. the input and output parameters are reasonable
        3. the dependency relationship between the nodes is correct
        4. whether some nodes can be merged or split

        please return the optimized json structure, keep the original format.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can optimize the workflow json structure."},
            {"role": "user", "content": optimization_prompt}
        ]
        try:
            response = await llm.single_generate_async(messages=messages, response_format={"type": "json_object"})
            # try to parse the LLM response
            optimized_json = json.loads(response)
            print("✅ LLM optimization finished")
            return optimized_json
        except Exception as e:
            print(f"❌ LLM optimization failed: {e}")
            return workflow_json

    def _instantiate_workflow(self, workflow_json: Dict[str, Any], llm) -> Any:
        """try to instantiate the workflow"""
        try:
            from ..workflow.workflow import WorkFlow
            from ..workflow.workflow_graph import WorkFlowGraph
            
            # create the workflow graph from the json
            graph = WorkFlowGraph.from_dict(workflow_json)

            # create the workflow instance
            workflow = WorkFlow(graph=graph, llm=llm)

            return workflow
        except Exception as e:
            logger.error(f"WorkFlow instantiation failed: {e}")
            raise


class HITLOutsideConversationAgent(HITLBaseAgent):
    """HITL Outside Conversation Agent - support the conversation loop to modify the workflow json structure"""

    def __init__(
        self,
        name: str = "HITLOutsideConversationAgent",
        description: str = "support the conversation loop to modify the workflow json structure",
        **kwargs
    ):
        super().__init__(
            name=name,
            description=description,
            is_human=True,
            **kwargs
        )

        # forbid the agent to be used in WorkFlow
        self.forbidden_in_workflow = True

        # add the conversation action
        action = HITLOutsideConversationAction()
        self.add_action(action)

    def get_hitl_agent_name(self) -> str:
        """get the HITL Agent name"""
        return self.name
    
    def execute(self, llm: BaseLLM, inputs: dict, hitl_manager: HITLManager, sys_msg: str = None, **kwargs) -> Tuple[dict, str]:
        """
        redirect to the HITLOutsideConversationAction.execute
        """
        if hasattr(self, 'actions') and len(self.actions) > 0:
            if isinstance(self.actions[0], HITLOutsideConversationAction):
                return self.actions[0].execute(llm, inputs, hitl_manager, sys_msg, **kwargs)
            else:
                raise ValueError(f"The first action of {self.name} must be HITLOutsideConversationAction, but got {self.actions[0].__class__}")
        else:
            raise ValueError(f"The {self.name} has no action")
        
    async def async_execute(self, llm: BaseLLM, inputs: dict, hitl_manager: HITLManager, sys_msg: str = None, **kwargs) -> Tuple[dict, str]:
        """
        redirect to the HITLOutsideConversationAction.async_execute
        """
        if hasattr(self, 'actions') and len(self.actions) > 0:
            if isinstance(self.actions[0], HITLOutsideConversationAction):
                return await self.actions[0].async_execute(llm, inputs, hitl_manager, sys_msg, **kwargs)
            else:
                raise ValueError(f"The first action of {self.name} must be HITLOutsideConversationAction, but got {self.actions[0].__class__}")
        else:
            raise ValueError(f"The {self.name} has no action")
        
    @property
    def conversation_action(self):
        """
        get the right conversation action
        """
        for action in self.actions:
            if isinstance(action, HITLOutsideConversationAction):
                return action
        raise ValueError(f"Action of class {HITLOutsideConversationAction.__name__} not found in {self}, please check the initialization of this Agent")