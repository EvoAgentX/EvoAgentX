import asyncio
import os
from typing import Optional
from pydantic import Field
from evoagentx.models.base_model import BaseLLM
from evoagentx.workflow import WorkFlow, WorkFlowGraph
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.agents import Agent, CustomizeAgent, AgentManager
from evoagentx.actions import Action, ActionInput, ActionOutput
from evoagentx.hitl import (
    HITLInterceptorAgent, 
    HITLInteractionType, 
    HITLMode,
    HITLManager
)
from evoagentx.models import OpenAILLMConfig, OpenAILLM 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EmailSendActionInput(ActionInput):
    human_verified_data: str = Field(description="The extracted data that will be sent to the user")

class EmailSendActionOuput(ActionOutput):
    send_action_result: str = Field(description="The sending action result")

class DummyEmailSendAction(Action):
    def __init__(
        self,
        name: str="EmailSendAction",
        description: str="A dummy action that send a email to use with extracted data",
        prompt: str="Send a email to the user with the extracted data",
        inputs_format: ActionInput=None,
        outputs_format: ActionOutput=None,
        **kwargs
    ):
        inputs_format: ActionInput = inputs_format or EmailSendActionInput
        outputs_format: ActionOutput = outputs_format or EmailSendActionOuput

        super().__init__(
            name=name,
            description=description,
            prompt=prompt,
            inputs_format=inputs_format,
            outputs_format=outputs_format,
            **kwargs
        )
    
    def execute(
        self, 
        llm: Optional[BaseLLM] = None, 
        inputs: Optional[dict] = None, 
        sys_msg: Optional[str] = None, 
        return_prompt: bool = False, 
        **kwargs
    ) -> EmailSendActionOuput:
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, "undefined") for attr in action_input_attrs}
        prompt = self.prompt.format(**action_input_data) # format the prompt with the action input data 
        
        # simulate the email sending process
        output: EmailSendActionOuput = EmailSendActionOuput(
            send_action_result=f"Email sent to user with extracted data: {action_input_data['human_verified_data']}"
        )
        if return_prompt:
            return output, prompt
        return output

    async def async_execute(
        self, 
        llm: Optional[BaseLLM] = None, 
        inputs: dict = None, 
        sys_msg: str = None, 
        return_prompt: bool = False, 
        **kwargs
    ) -> EmailSendActionOuput:
        return self.execute(llm, inputs, sys_msg, return_prompt, **kwargs)

async def main():
    print("🚀 EvoAgentX HITL example")
    print("=" * 60)

    llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)
    llm = OpenAILLM(llm_config)

    data_extraction_agent = CustomizeAgent(
        name="DataExtractionAgent",
        description="Extract numeric data from source",
        inputs=[{"name": "data_source", "type": "string", "description": "Source data location"}],
        outputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
        prompt="Extract data from source: {data_source}",
        llm_config=llm_config,
        parse_mode="str"
    )  

    dummy_data_sending_agent = Agent(
        name="DataSendingAgent",
        description="A dummy agent who sends the extracted data to the user by mail",
        actions=[DummyEmailSendAction()],
        llm_config=llm_config
    )
    
    # activate HITL feature
    hitl_manager = HITLManager()
    hitl_manager.activate()
    
    # create HITL interceptor agent
    data_interceptor = HITLInterceptorAgent(
        target_agent_name="DataSendingAgent",
        target_action_name="DummyEmailSendAction", 
        interaction_type=HITLInteractionType.APPROVE_REJECT,
        mode=HITLMode.PRE_EXECUTION
    )
    
    # node definition
    nodes = [
        WorkFlowNode(
            name="data_extraction_node",
            description="extract data from source",
            agents=[data_extraction_agent],
            inputs=[{"name": "data_source", "type": "string", "description": "Source data location"}],
            outputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}]
        ),
        WorkFlowNode(
            name="data_interceptor",
            description="intercept the execution of data sending agent",
            agents=[data_interceptor],
            inputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
            outputs=[{"name": "human_verified_data", "type": "string", "description": "Human verified data"}]
        ),
        WorkFlowNode(
            name="data_sending_node",
            description="send data to user by mail",
            agents=[dummy_data_sending_agent],
            inputs=[{"name": "human_verified_data", "type": "string", "description": "Human verified data"}],
            outputs=[{"name": "send_action_result", "type": "string", "description": "Send action result"}]
        )
    ]
    
    # edge definition
    edges = [
        WorkFlowEdge(source="data_extraction_node", target="data_interceptor"),
        WorkFlowEdge(source="data_interceptor", target="data_sending_node")
    ]
    
    # create workflow
    graph = WorkFlowGraph(
        goal="send the extracted data to the user by mail",
        nodes=nodes,
        edges=edges
    )
    
    agents = [data_extraction_agent, data_interceptor, dummy_data_sending_agent]

    # set up data field mapping inside the hitl agent
    hitl_data_mapping = {"human_verified_data": "extracted_data"}  # from outputs to inputs
    hitl_manager.hitl_input_output_mapping = hitl_data_mapping

    workflow = WorkFlow(graph=graph, llm=llm, agent_manager=AgentManager(agents=agents), hitl_manager=hitl_manager)
    
    # prepare the input data
    inputs = {
        "data_source": "In the second quarter of fiscal 2025, Aurora Technologies delivered robust financial performance, driven by sustained demand across its core semiconductor and software divisions. Total revenue reached $1.24 billion, marking an 8.4% year-over-year increase from $1.14 billion in Q2 2024. The semiconductor segment generated $780 million—up 10.1%—as sales of the new A7 microprocessor family accelerated, while the software and services division contributed $460 million, up 5.2%, on the strength of recurring cloud-based licensing revenues. Gross profit for the quarter was $520 million, yielding a gross margin of 41.9%, compared to $470 million and 41.2% margin in the year-ago period. Operating expenses totaled $310 million, comprising $130 million in research and development (10.5% of revenues) and $180 million in selling, general and administrative costs (14.5% of revenues). As a result, operating income improved to $210 million—an operating margin of 16.9%, up from $180 million and 15.8% last year. Net income attributable to shareholders was $165 million, or $0.82 per diluted share, reflecting an effective tax rate of 23.7%; this compares favorably with net income of $142 million, or $0.70 per share, in Q2 2024. Free cash flow for the quarter was $185 million, after capital expenditures of $45 million. The balance sheet remains strong, with cash and short-term investments of $720 million against total debt of $340 million, resulting in a net cash position of $380 million. In view of these results, management is raising full-year revenue guidance to $4.90–$5.00 billion (previously $4.80–$4.95 billion) and narrowing adjusted earnings-per-share guidance to $3.35–$3.45 (from $3.30–$3.50), reflecting confidence in continued growth across both semiconductor and software channels. The Board of Directors also approved a quarterly cash dividend of $0.15 per share, payable on July 15, 2025, to shareholders of record as of June 30, 2025."
    }
    
    try:
        print("\n📋 start to execute the workflow")
        result = await workflow.async_execute(inputs=inputs)
        print(f"\n✅ workflow executed successfully!")
        print(f"result: {result}")
        
    except Exception as e:
        print(f"\n❌ workflow execution failed: {e}")
    
    finally:
        # deactivate HITL feature
        hitl_manager.deactivate()

if __name__ == "__main__":
    asyncio.run(main())