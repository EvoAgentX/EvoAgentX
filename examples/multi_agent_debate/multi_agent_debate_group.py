from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.workflow.action_graph import ActionGraph
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.models.model_configs import OpenAILLMConfig
from typing import List


class GroupOfManyGraph(ActionGraph):
	name: str = "GroupOfManyGraph"
	description: str = "Group with variable number of inner debaters"
	llm_config: OpenAILLMConfig
	num_inner: int = 3

	_inner_debaters: List[CustomizeAgent] = None

	def init_module(self):
		super().init_module()
		prompt = (
			"""
You are a sub-team debater (role: {role}), agent {agent_id}, round {round_index}/{total_rounds}.
Problem:
{problem}

Transcript so far:
{transcript_text}

Return XML:
<response>
  <thought>...</thought>
  <argument>...</argument>
  <answer>optional</answer>
</response>
			"""
		).strip()

		self._inner_debaters = []
		for i in range(self.num_inner):
			debater = CustomizeAgent(
				name=f"GroupDebater#{i+1}",
				description="Inner debater of a group (variable size)",
				prompt=prompt,
				llm_config=self.llm_config,
				inputs=[
					{"name": "problem", "type": "str", "description": "The problem to debate"},
					{"name": "transcript_text", "type": "str", "description": "Transcript of previous debate rounds"},
					{"name": "role", "type": "str", "description": "Role of the debater"},
					{"name": "agent_id", "type": "str", "description": "Unique identifier for the agent"},
					{"name": "round_index", "type": "str", "description": "Current round number"},
					{"name": "total_rounds", "type": "str", "description": "Total number of debate rounds"},
				],
				outputs=[
					{"name": "thought", "type": "str", "required": True, "description": "The agent's reasoning process"},
					{"name": "argument", "type": "str", "required": True, "description": "The agent's main argument"},
					{"name": "answer", "type": "str", "required": False, "description": "Optional final answer from the agent"},
				],
				parse_mode="xml",
			)
			self._inner_debaters.append(debater)

	def execute(
		self,
		problem: str,
		transcript_text: str,
		role: str,
		agent_id: str,
		round_index: str,
		total_rounds: str,
		**kwargs
	) -> dict:
		arguments: List[str] = []
		thoughts: List[str] = []
		answers: List[str] = []

		local_transcript = transcript_text
		for i, debater in enumerate(self._inner_debaters):
			msg = debater(
				inputs=dict(
					problem=problem,
					transcript_text=local_transcript,
					role=f"{role} - #{i+1}",
					agent_id=f"{agent_id}_#{i+1}",
					round_index=round_index,
					total_rounds=total_rounds,
				)
			)
			data = msg.content.get_structured_data()
			arg = (data.get("argument", "") or "").strip()
			th = (data.get("thought", "") or "").strip()
			ans = (data.get("answer") or "").strip()

			arguments.append(f"[#{i+1}] {arg}")
			thoughts.append(f"[#{i+1}] {th}")
			if ans:
				answers.append(ans)

			local_transcript = local_transcript + "\n" + f"[GroupInner#{i+1} argument]: {arg}"

		answer = answers[-1] if answers else None
		argument_joined = "\n".join(arguments)
		thought_joined = " | ".join(thoughts)

		return {
			"argument": argument_joined,
			"answer": answer,
			"thought": thought_joined,
		}


if __name__ == "__main__":
	llm_cfg = OpenAILLMConfig(
		model="gpt-4o-mini",  # Replace with your locally available model name
		temperature=0.6,
		max_tokens=512,
	)

	group1 = GroupOfManyGraph(llm_config=llm_cfg, num_inner=3)
	group2 = GroupOfManyGraph(llm_config=llm_cfg, num_inner=4)
	group1.init_module()
	group2.init_module()
	group_graphs = [group1, group2]

	debate = MultiAgentDebateActionGraph(
		group_graphs_enabled=True,
		group_graphs=group_graphs,
		llm_config=llm_cfg,
	)
	debate.init_module()

	result = debate.execute(
		problem="Design a scalable multi-modal RAG system evaluation plan, including metrics and automated processes.",
		num_agents=2,
		num_rounds=3,
		judge_mode="llm_judge",
		return_transcript=True,
		enable_pruning=False,
	)

	print("WINNER:", result.get("winner"))
	print("FINAL ANSWER:\n", result["final_answer"]) 


