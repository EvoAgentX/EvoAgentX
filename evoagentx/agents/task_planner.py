from .agent import Agent
from ..actions.task_planning import TaskPlanning
from ..prompts.task_planner import TASK_PLANNER


class TaskPlanner(Agent):

    """
    An agent responsible for planning and decomposing high-level tasks into smaller sub-tasks.
    """
    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else TASK_PLANNER["name"]
        description = kwargs.pop("description") if "description" in kwargs else TASK_PLANNER["description"]
        system_prompt = kwargs.pop("system_prompt") if "system_prompt" in kwargs else TASK_PLANNER["system_prompt"]
        actions = kwargs.pop("actions") if "actions" in kwargs else [TaskPlanning()]
        super().__init__(name=name, description=description, system_prompt=system_prompt, actions=actions, **kwargs)
    
    @property
    def task_planning_action_name(self):
        return self.get_action_name(action_cls=TaskPlanning)
    