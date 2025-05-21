import random
import threading
from pydantic import Field
from typing import  Callable, Optional
from evoagentx.workflow import  WorkFlow
from evoagentx.agents import AgentManager
import tqdm

from evoagentx.workflow.workflow_graph import WorkFlowGraph

from .labeledfewshot import LabeledFewShot
from evoagentx.core.module import BaseModule
from evoagentx.core.registry import MODEL_REGISTRY 
from evoagentx.utils.mipro_utils.settings import settings
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.core.logging import logger

class BootstrapFewShot(BaseModule):
    """
    A Teleprompter class that composes a set of demos/examples to go into a agent's prompt.
    These demos come from a combination of labeled examples in the training set, and bootstrapped demos.
    """

    metric: Optional[Callable] = Field(
        default=None,
        description="A function that compares expected and predicted values, outputting the comparison result"
    )
    metric_threshold: Optional[float] = Field(
        default=None,
        description="If metric yields a numerical value, check against this threshold when deciding whether to accept a bootstrap example"
    )
    max_bootstrapped_demos: int = Field(
        default=4,
        description="Maximum number of bootstrapped demonstrations to include"
    )
    max_labeled_demos: int = Field(
        default=16,
        description="Maximum number of labeled demonstrations to include"
    )
    max_rounds: int = Field(
        default=1,
        description="Number of iterations to attempt generating required bootstrap examples. Program ends if exceeded"
    )
    max_errors: int = Field(
        default=5,
        description="Maximum number of errors until program ends"
    )

    def init_module(self):
        # Non-serializable runtime fields
        self.error_count = 0
        self.error_lock = threading.Lock()
        
    def optimize(self, student, *, teacher=None, trainset, collate_func):
        self.trainset = trainset
        self.collate_func = collate_func

        self._prepare_student_and_teacher(student, teacher)
        self._prepare_agent_mappings()
        self._bootstrap()

        self.student = self._train()
        self.student._compiled = True

        # set assert_failures and suggest_failures as attributes of student w/ value 0
        self.student._assert_failures = 0
        self.student._suggest_failures = 0

        return self.student
    
    def _prepare_student_and_teacher(self, student, teacher):
        self.student = student.reset_copy()
        self.teacher = teacher.deep_copy() if teacher is not None else student.deep_copy()
        for agent in self.teacher.agents():
            agent.setdefault("demos", [])
            agent.setdefault("traces", [])
            agent.setdefault("train", [])
        
        assert getattr(self.student, "_compiled", False) is False, "Student must be uncompiled."

        if self.max_labeled_demos and getattr(self.teacher, "_compiled", False) is False:
            optimizer = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = optimizer.optimize(self.teacher.reset_copy(), trainset=self.trainset)
        
    def _prepare_agent_mappings(self):
        name2agent, agent2name = {}, {}
        student, teacher = self.student, self.teacher
        assert len(student.agents()) == len(
            teacher.agents(),
        ), "Student and teacher must have the same number of agents."

        for agent1, agent2 in zip(student.agents(), teacher.agents()):
            name1 = agent1['name']
            name2 = agent2['name']
            assert name1 == name2, "Student and teacher must have the same program structure."

            name2agent[name1] = None  # dict(student=agent1, teacher=agent2)
            
            agent2name[id(agent1)] = name1
            agent2name[id(agent2)] = name2

        self.name2agent = name2agent
        self.agent2name = agent2name
        
    def _bootstrap(self, *, max_bootstraps=None):
        max_bootstraps = max_bootstraps or self.max_bootstrapped_demos
        bootstrap_attempts = 0

        bootstrapped = {}
        self.name2traces = {name: [] for name in self.name2agent}

        for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
            if len(bootstrapped) >= max_bootstraps:
                break

            for round_idx in range(self.max_rounds):
                bootstrap_attempts += 1

                if self._bootstrap_one_example(example, round_idx):
                    bootstrapped[example_idx] = True
                    break

        print(
            f"Bootstrapped {len(bootstrapped)} full traces after {example_idx} examples "
            f"for up to {self.max_rounds} rounds, amounting to {bootstrap_attempts} attempts."
        )

        # Unbootstrapped training examples

        self.validation = [x for idx, x in enumerate(self.trainset) if idx not in bootstrapped]
        random.Random(0).shuffle(self.validation)

        self.validation = self.validation

        # NOTE: Can't yet use evaluate because we need to trace *per example*
        # evaluate = Evaluate(program=self.teacher, metric=self.metric, num_threads=12)
        # score = evaluate(self.metric, display_table=False, display_progress=True)

    def _bootstrap_one_example(self, example, round_idx=0):
        name2traces = {}
        teacher = self.teacher
        agent_cache = {}
        try:

            # llm_config  = (settings.executor_llm or settings.llm_config).copy()
            llm_config = getattr(settings, 'optimizer_llm', settings.llm_config).copy()
            if round_idx > 0:
                llm_config.temperature = 0.7 + 0.001 * round_idx
                
            for agent in teacher.agents():
                name = agent['name']
                
                agent_cache[name] = agent['demos']
                agent['demos'] = [x for x in agent['demos'] if x != example]
            
            agent_manager = AgentManager()
            agent_manager.clear_agents()
            agent_manager.add_agents_from_workflow(
                teacher,
                llm_config= llm_config
            )
            
            program_copy = WorkFlowGraph(goal=teacher.goal, graph=teacher.graph)
            program_copy.reset_graph()
            cls = MODEL_REGISTRY.get_model(llm_config.llm_type)
            workflow = WorkFlow(graph=teacher, agent_manager= agent_manager, llm=cls(llm_config)) # llm=OpenAILLM(lm))
            
            with suppress_logger_info():
                prediction = workflow.execute(inputs = self.collate_func(example))
            
            trace = settings.trace
            for agent in teacher.agents():
                agent['demos'] = agent_cache[agent['name']]
            if self.metric:
                metric_val = self.metric(example, prediction, trace)
                if self.metric_threshold:
                    success = metric_val >= self.metric_threshold
                else:
                    success = metric_val
            else:
                success = True
        except Exception as e:
            success = False
            with self.error_lock:
                self.error_count += 1
                current_error_count = self.error_count
            if current_error_count >= self.max_errors:
                raise e
            logger.error(f"Failed to run or to evaluate example with {self.metric} due to {e}.")

        if success:
            for step in trace:
                agent, inputs, outputs = step
                demo = {"augmented": True, **inputs, **outputs}

                try:
                    agent_name = self.agent2name[id(agent)]
                except KeyError:
                    continue  # FIXME: !

                    # # TODO: Look closer into this. It's a bit tricky to reproduce.
                    # print(f"Failed to find agent {agent} in {self.agent2name}.")
                    # print(
                    #     "Are you doing this in a notebook (Jupyter)? This might be caused by redefining values by rerunning cells.",
                    # )
                    # print("Try restarting the notebook, or open an issue.")
                    # raise KeyError(
                    #     f"Failed to find agent {id(agent)} {agent} in {self.agent2name}.",
                    # ) from e

                name2traces[agent_name] = name2traces.get(agent_name, [])
                name2traces[agent_name].append(demo)

            # Update the traces
            for name, demos in name2traces.items():
                from datasets.fingerprint import Hasher

                # If there are multiple traces for the same agent in the sample example,
                # sample 50/50 from the first N-1 traces or the last trace.
                if len(demos) > 1:
                    rng = random.Random(Hasher.hash(tuple(demos)))
                    demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]
                self.name2traces[name].extend(demos)

        return success
    
    def _train(self):
        rng = random.Random(0)
        raw_demos = self.validation
        
        for agent in self.student.agents():
            name = agent['name']
            
            augmented_demos = self.name2traces[name][:self.max_labeled_demos]
            
            sample_size = min(self.max_labeled_demos - len(augmented_demos), len(raw_demos))
            sample_size = max(sample_size, 0)
            
            raw_demos = rng.sample(raw_demos, sample_size)
            agent['demos'] = augmented_demos + raw_demos
        return self.student

