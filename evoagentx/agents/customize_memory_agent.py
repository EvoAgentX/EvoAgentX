from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from pydantic import Field, create_model

from ..actions.action import Action, ActionInput, ActionOutput
from ..actions.customize_action import CustomizeAction
from ..core.logging import logger
from ..core.message import Message, MessageType
from ..core.module import BaseModule
from ..core.registry import MODULE_REGISTRY, PARSE_FUNCTION_REGISTRY
from ..models.model_configs import LLMConfig
from ..models.base_model import PARSER_VALID_MODE
from ..prompts.template import PromptTemplate
from ..prompts.utils import DEFAULT_SYSTEM_PROMPT
from ..storages.base import StorageHandler
from ..tools.tool import Tool, Toolkit
from ..utils.utils import generate_dynamic_class_name, make_parent_folder
from .agent import Agent


class MemoryOperation(str, Enum):
    """Canonical lifecycle stages for unified memory orchestration."""

    RETRIEVE = "retrieve"
    INJECT = "inject"
    EXECUTE = "execute"
    REFLECT = "reflect"
    PERSIST = "persist"


@dataclass
class MemoryRunContext:
    """
    Lightweight container tracking state as a run moves through the memory pipeline.

    Attributes:
        action_name: Name of the action being executed.
        action: Action instance bound to the agent.
        inputs: Mutable dictionary of action inputs (post injection).
        messages: Optional conversation history passed to the run.
        query: Query string used for retrieval (may be inferred).
        top_k: Retrieval fan-out cap.
        retrieved: Mapping of backend name to retrieval payloads.
        injected_blocks: Ordered list of prompt/context artifacts produced during injection.
        execution_result: Final Message returned by the action stage.
        reflection_outputs: Backend specific reflection payloads.
        persistence_results: Backend specific persist outcomes.
        metadata: Scratchpad for orchestration-specific markers.
    """

    action_name: str
    action: Action
    inputs: Dict[str, Any]
    messages: Optional[List[Message]] = None
    query: Optional[Any] = None
    top_k: int = 3
    retrieved: Dict[str, Any] = field(default_factory=dict)
    injected_blocks: List[Any] = field(default_factory=list)
    execution_result: Optional[Message] = None
    reflection_outputs: Dict[str, Any] = field(default_factory=dict)
    persistence_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMemoryBackend(BaseModule):
    """
    Abstract base class for pluggable memory backends.

    Backends may represent vector stores, structured knowledge bases, workflow archives,
    or behavioural traces. They are expected to implement one or more lifecycle hooks
    defined by :class:`MemoryOperation`.

    Implementations should override the relevant ``on_<operation>`` methods. Each method
    receives the running agent and :class:`MemoryRunContext`, enabling full access to the
    evolving state.
    """

    name: str = Field(default="memory-backend")
    priority: int = Field(default=100, description="Lower value -> higher precedence during orchestration.")
    supported_operations: Tuple[MemoryOperation, ...] = Field(
        default=(
            MemoryOperation.RETRIEVE,
            MemoryOperation.INJECT,
            MemoryOperation.REFLECT,
            MemoryOperation.PERSIST,
        ),
        description="Lifecycle operations implemented by this backend.",
    )

    def supports(self, operation: MemoryOperation) -> bool:
        return operation in self.supported_operations

    async def dispatch(
        self,
        operation: MemoryOperation,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
        **kwargs: Any,
    ) -> Any:
        """
        Execute the backend handler for ``operation`` if available.

        Returns ``None`` when the backend does not implement the operation or the handler
        intentionally yields no artifact.
        """
        handler_name = f"on_{operation.value}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            return None
        result = handler(agent=agent, run_context=run_context, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    # Optional lifecycle handlers -------------------------------------------------
    def on_retrieve(
        self,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
        query: str,
        top_k: int,
        **_: Any,
    ) -> Any:
        """Override to return backend-specific retrieval payload."""
        raise NotImplementedError

    def on_inject(
        self,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
        retrieved: Any,
        **_: Any,
    ) -> Any:
        """
        Override to convert retrieval payloads into prompt/context fragments.
        Should return an artifact (e.g., str or dict) that downstream stages can consume.
        """
        return None

    def on_reflect(
        self,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
        execution_result: Message,
        **_: Any,
    ) -> Any:
        """Override to perform summarisation/extraction based on execution outcome."""
        return None

    def on_persist(
        self,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
        reflection: Any,
        **_: Any,
    ) -> Any:
        """Override to store reflection artifacts back to the backend."""
        return None


class BaseMemoryPolicy(BaseModule):
    """
    Policies coordinate how backends participate in each lifecycle stage.
    They can dynamically filter, order, or transform backend outputs.
    """

    name: str = Field(default="default-policy")
    default_top_k: int = Field(default=3)

    def select_backends(
        self,
        operation: MemoryOperation,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
    ) -> Sequence[BaseMemoryBackend]:
        """Select participating backends for ``operation``."""
        del run_context  # unused by default policy
        eligible = [
            backend
            for backend in agent.memory_backends.values()
            if backend.supports(operation)
        ]
        return sorted(eligible, key=lambda b: b.priority)

    def prepare_query(
        self,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
        **kwargs: Any,
    ) -> Optional[Any]:
        """
        Determine the retrieval query string.

        Default behaviour prefers explicitly provided query -> run context query ->
        newest user message content -> empty.
        """
        query = kwargs.get("memory_query") or run_context.query
        if query:
            return query
        if run_context.inputs:
            for key in ("query", "prompt", "user_input", "task"):
                candidate = run_context.inputs.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate
        if run_context.messages:
            for message in reversed(run_context.messages):
                if isinstance(message.content, str) and message.content.strip():
                    return message.content
        return None

    def aggregate_injections(
        self,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
        injections: Sequence[Any],
    ) -> Dict[str, Any]:
        """
        Combine backend injections into the final action inputs.

        Default concatenates string blocks and stores them under ``memory_context``.
        """
        text_blocks = [block for block in injections if isinstance(block, str)]
        structured_blocks = [block for block in injections if not isinstance(block, str)]

        if text_blocks:
            agent_block = "\n\n".join(text_blocks)
            run_context.inputs.setdefault("memory_context", agent_block)

        if structured_blocks:
            run_context.inputs.setdefault("memory_artifacts", []).extend(structured_blocks)

        return run_context.inputs

    def should_reflect(self, agent: "CustomizeMemoryAgent", run_context: MemoryRunContext) -> bool:
        """Determine whether reflection should be executed."""
        return agent.reflection_enabled


class DefaultMemoryPolicy(BaseMemoryPolicy):
    """Thin wrapper to keep naming explicit for default policy resolution."""


class MemoryOrchestrator(BaseModule):
    """
    Controls the lifecycle ordering for each run.

    Subclasses can override :meth:`plan` or :meth:`after_stage` to implement adaptive
    orchestration (e.g., skipping persist on read-only requests, branching on errors).
    """

    pipeline: Tuple[MemoryOperation, ...] = Field(
        default=(
            MemoryOperation.RETRIEVE,
            MemoryOperation.INJECT,
            MemoryOperation.EXECUTE,
            MemoryOperation.REFLECT,
            MemoryOperation.PERSIST,
        )
    )

    def plan(self, agent: "CustomizeMemoryAgent", run_context: MemoryRunContext) -> Sequence[MemoryOperation]:
        del agent, run_context  # default ignores inputs
        return self.pipeline

    def after_stage(
        self,
        agent: "CustomizeMemoryAgent",
        run_context: MemoryRunContext,
        operation: MemoryOperation,
        result: Any,
    ) -> None:
        """Optional hook triggered after each stage completes."""
        del agent, run_context, result, operation


class CustomizeMemoryAgent(Agent):
    """
    A customizable agent with modular, pluggable memory backends.

    The agent exposes a unified lifecycle:

        retrieve -> inject -> execute -> reflect -> persist

    Memory backends can be declared via config dictionaries (using ``class_name``)
    or provided as instantiated :class:`BaseMemoryBackend` objects.
    """

    memory_backends: Dict[str, BaseMemoryBackend] = Field(default_factory=dict)
    memory_orchestrator: Optional[MemoryOrchestrator] = None
    memory_policies: Dict[str, BaseMemoryPolicy] = Field(default_factory=dict)
    storage_handler: Optional[StorageHandler] = None
    current_context: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    reflection_enabled: bool = Field(default=True)

    def __init__(
        self,
        name: str,
        description: str,
        prompt: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        llm_config: Optional[LLMConfig] = None,
        inputs: Optional[List[dict]] = None,
        outputs: Optional[List[dict]] = None,
        system_prompt: Optional[str] = None,
        output_parser: Optional[Type[ActionOutput]] = None,
        parse_mode: Optional[str] = "title",
        parse_func: Optional[Callable] = None,
        title_format: Optional[str] = None,
        tools: Optional[List[Union[Toolkit, Tool]]] = None,
        max_tool_calls: Optional[int] = 5,
        custom_output_format: Optional[str] = None,
        memory_backends: Optional[Dict[str, BaseMemoryBackend]] = None,
        memory_orchestrator: Optional[MemoryOrchestrator] = None,
        memory_policies: Optional[Dict[str, BaseMemoryPolicy]] = None,
        storage_handler: Optional[StorageHandler] = None,
        **kwargs: Any,
    ):
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        inputs = inputs or []
        outputs = outputs or []

        if tools is not None:
            raw_tool_map = {tool.name: tool for tool in tools}
            tools = [tool if isinstance(tool, Toolkit) else Toolkit(name=tool.name, tools=[tool]) for tool in tools]
        else:
            raw_tool_map = None

        if prompt is not None and prompt_template is not None:
            logger.warning(
                "Both `prompt` and `prompt_template` are provided in `CustomizeMemoryAgent`. "
                "`prompt_template` will be used."
            )
            prompt = None

        if isinstance(parse_func, str):
            if not PARSE_FUNCTION_REGISTRY.has_function(parse_func):
                raise ValueError(
                    f"parse function `{parse_func}` is not registered! "
                    "Use decorator `@register_parse_function` to register the parse function."
                )
            parse_func = PARSE_FUNCTION_REGISTRY.get_function(parse_func)

        if isinstance(output_parser, str):
            output_parser = MODULE_REGISTRY.get_module(output_parser)

        if parse_mode == "title" and title_format is None:
            title_format = "## {title}"

        self.validate_data(
            prompt=prompt,
            prompt_template=prompt_template,
            inputs=inputs,
            outputs=outputs,
            output_parser=output_parser,
            parse_mode=parse_mode,
            parse_func=parse_func,
            title_format=title_format,
        )

        memory_backends = dict(memory_backends or kwargs.pop("memory_backends", None) or {})
        memory_policies = dict(memory_policies or kwargs.pop("memory_policies", None) or {})
        memory_orchestrator = memory_orchestrator or kwargs.pop("memory_orchestrator", None)

        customize_action = self.create_customize_action(
            name=name,
            desc=description,
            prompt=prompt,
            prompt_template=prompt_template,
            inputs=inputs,
            outputs=outputs,
            parse_mode=parse_mode,
            parse_func=parse_func,
            output_parser=output_parser,
            title_format=title_format,
            custom_output_format=custom_output_format,
            tools=tools,
            max_tool_calls=max_tool_calls,
        )

        super().__init__(
            name=name,
            description=description,
            llm_config=llm_config,
            system_prompt=system_prompt,
            storage_handler=storage_handler,
            actions=[customize_action],
            memory_backends=memory_backends,
            memory_orchestrator=memory_orchestrator,
            memory_policies=memory_policies,
            **kwargs,
        )
        self._store_inputs_outputs_info(inputs, outputs, raw_tool_map)
        self.output_parser = output_parser
        self.parse_mode = parse_mode
        self.parse_func = parse_func
        self.title_format = title_format
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.custom_output_format = custom_output_format

    def _add_tools(self, tools: List[Toolkit]):
        self.get_action(self.customize_action_name).add_tools(tools)

    @property
    def customize_action_name(self) -> str:
        """
        Get the name of the primary custom action for this agent.
        """
        for action in self.actions:
            if action.name != self.cext_action_name:
                return action.name
        raise ValueError("Couldn't find the customize action name!")

    @property
    def action(self) -> Action:
        """Get the primary custom action."""
        return self.get_action(self.customize_action_name)

    @property
    def prompt(self) -> str:
        """Get the prompt for the primary custom action."""
        return self.action.prompt

    @property
    def prompt_template(self) -> PromptTemplate:
        """Get the prompt template for the primary custom action."""
        return self.action.prompt_template

    def validate_data(
        self,
        prompt: Optional[str],
        prompt_template: Optional[PromptTemplate],
        inputs: List[dict],
        outputs: List[dict],
        output_parser: Optional[Type[ActionOutput]],
        parse_mode: str,
        parse_func: Optional[Callable],
        title_format: Optional[str],
    ):
        if prompt is None and prompt_template is None:
            raise ValueError("`prompt` or `prompt_template` is required when creating a CustomizeMemoryAgent.")

        if prompt_template is None and inputs:
            all_input_names = [input_item["name"] for input_item in inputs]
            inputs_names_not_in_prompt = [name for name in all_input_names if f"{{{name}}}" not in (prompt or "")]
            if inputs_names_not_in_prompt:
                raise KeyError(f"The following inputs are not found in the prompt: {inputs_names_not_in_prompt}.")

        if output_parser is not None:
            self._check_output_parser(outputs, output_parser)

        if parse_mode not in PARSER_VALID_MODE:
            raise ValueError(f"'{parse_mode}' is an invalid value for `parse_mode`. Available choices: {PARSER_VALID_MODE}.")

        if parse_mode == "custom":
            if parse_func is None:
                raise ValueError(
                    "`parse_func` (a callable function with an input argument `content`) must be provided when "
                    "`parse_mode` is 'custom'."
                )

        if parse_func is not None:
            if not callable(parse_func):
                raise ValueError("`parse_func` must be a callable function with an input argument `content`.")
            signature = inspect.signature(parse_func)
            if "content" not in signature.parameters:
                raise ValueError("`parse_func` must have an input argument `content`.")
            if not PARSE_FUNCTION_REGISTRY.has_function(parse_func.__name__):
                logger.warning(
                    f"parse function `{parse_func.__name__}` is not registered. This can cause issues when loading the agent "
                    "from a file. It is recommended to register the parse function using `register_parse_function`."
                )

        if parse_mode == "title" and title_format is not None and "{title}" not in title_format:
            raise ValueError(r"`title_format` must contain the placeholder `{title}`.")

    def create_customize_action(
        self,
        name: str,
        desc: str,
        prompt: Optional[str],
        prompt_template: Optional[PromptTemplate],
        inputs: List[dict],
        outputs: List[dict],
        parse_mode: str,
        parse_func: Optional[Callable] = None,
        output_parser: Optional[ActionOutput] = None,
        title_format: Optional[str] = "## {title}",
        custom_output_format: Optional[str] = None,
        tools: Optional[List[Toolkit]] = None,
        max_tool_calls: Optional[int] = 5,
    ) -> Action:
        assert prompt is not None or prompt_template is not None, (
            "must provide `prompt` or `prompt_template` when creating CustomizeMemoryAgent"
        )

        action_input_fields = {}
        for field in inputs:
            required = field.get("required", True)
            if required:
                action_input_fields[field["name"]] = (str, Field(description=field["description"]))
            else:
                action_input_fields[field["name"]] = (
                    Optional[str],
                    Field(default=None, description=field["description"]),
                )

        action_input_type = create_model(
            self._get_unique_class_name(generate_dynamic_class_name(name + " action_input")),
            **action_input_fields,
            __base__=ActionInput,
        )

        if output_parser is None:
            action_output_fields = {}
            for field in outputs:
                required = field.get("required", True)
                if required:
                    action_output_fields[field["name"]] = (Any, Field(description=field["description"]))
                else:
                    action_output_fields[field["name"]] = (
                        Optional[Any],
                        Field(default=None, description=field["description"]),
                    )
            action_output_type = create_model(
                self._get_unique_class_name(generate_dynamic_class_name(name + " action_output")),
                **action_output_fields,
                __base__=ActionOutput,
            )
        else:
            action_output_type = output_parser

        action_cls_name = self._get_unique_class_name(generate_dynamic_class_name(name + " action"))

        customize_action_cls = create_model(action_cls_name, __base__=CustomizeAction)

        customize_action = customize_action_cls(
            name=action_cls_name,
            description=desc,
            prompt=prompt,
            prompt_template=prompt_template,
            inputs_format=action_input_type,
            outputs_format=action_output_type,
            parse_mode=parse_mode,
            parse_func=parse_func,
            title_format=title_format,
            custom_output_format=custom_output_format,
            max_tool_try=max_tool_calls,
            tools=tools,
        )
        return customize_action

    def _check_output_parser(self, outputs: List[dict], output_parser: Type[ActionOutput]):
        if output_parser is not None:
            if not isinstance(output_parser, type):
                raise TypeError(f"output_parser must be a class, but got {type(output_parser).__name__}")
            if not issubclass(output_parser, ActionOutput):
                raise ValueError(
                    f"`output_parser` must be a subclass of `ActionOutput`, but got `{output_parser.__name__}`."
                )

            output_parser_fields = output_parser.get_attrs()
            all_output_names = [output_item["name"] for output_item in outputs]
            for field in output_parser_fields:
                if field not in all_output_names:
                    raise ValueError(
                        f"The output parser `{output_parser.__name__}` is not compatible with the `outputs`.\n"
                        f"The output parser fields: {output_parser_fields}.\n"
                        f"The outputs: {all_output_names}.\n"
                        f"All the fields in the output parser must be present in the outputs."
                    )

    def _store_inputs_outputs_info(
        self,
        inputs: List[dict],
        outputs: List[dict],
        tool_map: Optional[Dict[str, Union[Toolkit, Tool]]],
    ):
        self._action_input_types, self._action_input_required = {}, {}
        for field in inputs:
            required = field.get("required", True)
            self._action_input_types[field["name"]] = field["type"]
            self._action_input_required[field["name"]] = required
        self._action_output_types, self._action_output_required = {}, {}
        for field in outputs:
            required = field.get("required", True)
            self._action_output_types[field["name"]] = field["type"]
            self._action_output_required[field["name"]] = required
        self._raw_tool_map = tool_map

    def __call__(
        self,
        inputs: Optional[dict] = None,
        return_msg_type: MessageType = MessageType.UNKNOWN,
        **kwargs: Any,
    ) -> Message:
        inputs = inputs or {}
        return super().__call__(
            action_name=self.customize_action_name,
            action_input_data=inputs,
            return_msg_type=return_msg_type,
            **kwargs,
        )

    def get_customize_agent_info(self) -> dict:
        customize_action = self.get_action(self.customize_action_name)
        action_input_params = customize_action.inputs_format.get_attrs()
        action_output_params = customize_action.outputs_format.get_attrs()

        config = {
            "class_name": "CustomizeMemoryAgent",
            "name": self.name,
            "description": self.description,
            "prompt": customize_action.prompt,
            "prompt_template": customize_action.prompt_template.to_dict()
            if customize_action.prompt_template is not None
            else None,
            "inputs": [
                {
                    "name": field,
                    "type": self._action_input_types[field],
                    "description": field_info.description,
                    "required": self._action_input_required[field],
                }
                for field, field_info in customize_action.inputs_format.model_fields.items()
                if field in action_input_params
            ],
            "outputs": [
                {
                    "name": field,
                    "type": self._action_output_types[field],
                    "description": field_info.description,
                    "required": self._action_output_required[field],
                }
                for field, field_info in customize_action.outputs_format.model_fields.items()
                if field in action_output_params
            ],
            "system_prompt": self.system_prompt,
            "output_parser": self.output_parser.__name__ if self.output_parser is not None else None,
            "parse_mode": self.parse_mode,
            "parse_func": self.parse_func.__name__ if self.parse_func is not None else None,
            "title_format": self.title_format,
            "tool_names": [tool.name for tool in customize_action.tools] if customize_action.tools else [],
            "max_tool_calls": self.max_tool_calls,
            "custom_output_format": self.custom_output_format,
        }
        return config

    @classmethod
    def load_module(
        cls,
        path: str,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[Union[Toolkit, Tool]]] = None,
        **kwargs: Any,
    ) -> dict:
        match_dict: Dict[str, Union[Toolkit, Tool]] = {}
        agent = super().load_module(path=path, llm_config=llm_config, **kwargs)
        if tools:
            match_dict = {tool.name: tool for tool in tools}
        if agent.get("tool_names"):
            assert (
                tools is not None
            ), "must provide `tools: List[Union[Toolkit, Tool]]` when loading the agent and `tool_names` is not empty"
            added_tools = [match_dict[tool_name] for tool_name in agent["tool_names"]]
            agent["tools"] = [
                tool if isinstance(tool, Toolkit) else Toolkit(name=tool.name, tools=[tool]) for tool in added_tools
            ]
        return agent

    def save_module(self, path: str, ignore: List[str] = None, **kwargs: Any) -> str:
        ignore = ignore or []
        config = self.get_customize_agent_info()
        for ignore_key in ignore:
            config.pop(ignore_key, None)
        make_parent_folder(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return path

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

    def get_config(self) -> dict:
        config = self.get_customize_agent_info()
        if self.llm_config:
            config["llm_config"] = self.llm_config.to_dict()
        tool_names = config.pop("tool_names", None)
        if tool_names and self._raw_tool_map:
            config["tools"] = [self._raw_tool_map[name] for name in tool_names]
        return config
    # --------------------------------------------------------------------- init --
    def init_module(self):
        super().init_module()

        if not self.memory_policies:
            self.memory_policies = {"default": DefaultMemoryPolicy()}
        else:
            self.memory_policies = {
                key: self._ensure_policy_instance(value) for key, value in self.memory_policies.items()
            }

        self._default_policy = self.memory_policies.get("default") or DefaultMemoryPolicy()

        if self.memory_orchestrator is None:
            self.memory_orchestrator = MemoryOrchestrator()
        else:
            self.memory_orchestrator = self._ensure_orchestrator_instance(self.memory_orchestrator)

        # Ensure backends are instantiated and registered.
        self.memory_backends = {
            name: self._ensure_backend_instance(name, backend)
            for name, backend in self.memory_backends.items()
        }

        self.current_context = {}
        self.execution_trace = []

    # -------------------------------------------------------------- registration --
    def register_backend(self, backend: BaseMemoryBackend, name: Optional[str] = None) -> None:
        backend_name = name or getattr(backend, "name", backend.class_name)
        if backend_name in self.memory_backends:
            logger.warning("Overwriting existing memory backend '%s'", backend_name)
        self.memory_backends[backend_name] = backend
        logger.info("[CustomizeMemoryAgent] Registered memory backend '%s'", backend_name)

    def unregister_backend(self, name: str) -> None:
        if name in self.memory_backends:
            self.memory_backends.pop(name)
            logger.info("[CustomizeMemoryAgent] Unregistered memory backend '%s'", name)

    def get_backend(self, name: str) -> BaseMemoryBackend:
        if name not in self.memory_backends:
            raise KeyError(f"Memory backend '{name}' is not registered.")
        return self.memory_backends[name]

    # ---------------------------------------------------------------- lifecycle --
    async def async_execute(
        self,
        action_name: str,
        msgs: Optional[List[Message]] = None,
        action_input_data: Optional[dict] = None,
        return_msg_type: MessageType = MessageType.UNKNOWN,
        return_action_input_data: Optional[bool] = False,
        **kwargs: Any,
    ):
        return await self._run_with_memory(
            action_name=action_name,
            msgs=msgs,
            action_input_data=action_input_data,
            return_msg_type=return_msg_type,
            return_action_input_data=return_action_input_data,
            is_async=True,
            **kwargs,
        )

    def execute(
        self,
        action_name: str,
        msgs: Optional[List[Message]] = None,
        action_input_data: Optional[dict] = None,
        return_msg_type: MessageType = MessageType.UNKNOWN,
        return_action_input_data: Optional[bool] = False,
        **kwargs: Any,
    ):
        return asyncio.run(
            self._run_with_memory(
                action_name=action_name,
                msgs=msgs,
                action_input_data=action_input_data,
                return_msg_type=return_msg_type,
                return_action_input_data=return_action_input_data,
                is_async=False,
                **kwargs,
            )
        )

    async def _run_with_memory(
        self,
        action_name: str,
        msgs: Optional[List[Message]],
        action_input_data: Optional[dict],
        return_msg_type: MessageType,
        return_action_input_data: bool,
        is_async: bool,
        **kwargs: Any,
    ):
        action, base_inputs = self._prepare_execution(
            action_name=action_name,
            msgs=msgs,
            action_input_data=action_input_data,
            **kwargs,
        )

        metadata = kwargs.pop("memory_metadata", None)
        run_context = MemoryRunContext(
            action_name=action_name,
            action=action,
            inputs=base_inputs or {},
            messages=msgs,
            top_k=kwargs.get("memory_top_k", None) or self._default_policy.default_top_k,
        )
        if metadata:
            run_context.metadata.update(metadata)

        await self.before_run(run_context=run_context, **kwargs)

        pipeline = self.memory_orchestrator.plan(agent=self, run_context=run_context)
        for operation in pipeline:
            if operation is MemoryOperation.RETRIEVE:
                await self._stage_retrieve(run_context, **kwargs)
            elif operation is MemoryOperation.INJECT:
                await self._stage_inject(run_context, **kwargs)
            elif operation is MemoryOperation.EXECUTE:
                await self._stage_execute(
                    run_context=run_context,
                    return_msg_type=return_msg_type,
                    is_async=is_async,
                    **kwargs,
                )
            elif operation is MemoryOperation.REFLECT:
                await self._stage_reflect(run_context, **kwargs)
            elif operation is MemoryOperation.PERSIST:
                await self._stage_persist(run_context, **kwargs)
            else:
                logger.warning("Unknown memory operation: %s", operation)
                continue

        await self.after_run(run_context=run_context, **kwargs)

        result_message = run_context.execution_result
        if return_action_input_data:
            return result_message, run_context.inputs
        return result_message

    # ---------------------------------------------------------- stage handlers --
    async def _stage_retrieve(self, run_context: MemoryRunContext, **kwargs: Any) -> None:
        policy = self._get_policy(MemoryOperation.RETRIEVE)
        query = policy.prepare_query(agent=self, run_context=run_context, **kwargs)
        run_context.query = query

        if not query:
            logger.debug("[CustomizeMemoryAgent] Retrieval skipped (empty query).")
            return

        backends = policy.select_backends(
            operation=MemoryOperation.RETRIEVE,
            agent=self,
            run_context=run_context,
        )

        retrievals = {}
        for backend in backends:
            try:
                result = await backend.dispatch(
                    MemoryOperation.RETRIEVE,
                    agent=self,
                    run_context=run_context,
                    query=query,
                    top_k=run_context.top_k,
                    **kwargs,
                )
                if result is not None:
                    retrievals[backend.name] = result
            except Exception as exc:
                logger.exception(
                    "[CustomizeMemoryAgent] Backend '%s' retrieval failed: %s",
                    backend.name,
                    exc,
                )

        run_context.retrieved = retrievals
        self._record_trace(run_context, operation=MemoryOperation.RETRIEVE, payload=retrievals)

    async def _stage_inject(self, run_context: MemoryRunContext, **kwargs: Any) -> None:
        if not run_context.retrieved:
            logger.debug("[CustomizeMemoryAgent] Injection skipped (no retrieval payloads).")
            return

        policy = self._get_policy(MemoryOperation.INJECT)
        backends = policy.select_backends(
            operation=MemoryOperation.INJECT,
            agent=self,
            run_context=run_context,
        )

        injections = []
        for backend in backends:
            payload = run_context.retrieved.get(backend.name)
            if payload is None:
                continue
            try:
                block = await backend.dispatch(
                    MemoryOperation.INJECT,
                    agent=self,
                    run_context=run_context,
                    retrieved=payload,
                    **kwargs,
                )
                if block:
                    injections.append(block)
            except Exception as exc:
                logger.exception(
                    "[CustomizeMemoryAgent] Backend '%s' injection failed: %s",
                    backend.name,
                    exc,
                )

        run_context.injected_blocks.extend(injections)
        policy.aggregate_injections(agent=self, run_context=run_context, injections=injections)
        self._record_trace(run_context, operation=MemoryOperation.INJECT, payload=injections)

    async def _stage_execute(
        self,
        run_context: MemoryRunContext,
        return_msg_type: MessageType,
        is_async: bool,
        **kwargs: Any,
    ) -> None:
        action = run_context.action
        inputs = run_context.inputs
        msg_type = return_msg_type or MessageType.UNKNOWN

        async_execute_source = inspect.getsource(action.async_execute)
        try:
            if "NotImplementedError" in async_execute_source or not is_async:
                execution_results = action.execute(
                    llm=self.llm,
                    inputs=inputs,
                    sys_msg=self.system_prompt,
                    return_prompt=True,
                    **kwargs,
                )
            else:
                execution_results = await action.async_execute(
                    llm=self.llm,
                    inputs=inputs,
                    sys_msg=self.system_prompt,
                    return_prompt=True,
                    **kwargs,
                )
        except NotImplementedError:
            execution_results = action.execute(
                llm=self.llm,
                inputs=inputs,
                sys_msg=self.system_prompt,
                return_prompt=True,
                **kwargs,
            )

        action_output, prompt = execution_results
        message = self._create_output_message(
            action_output=action_output,
            prompt=prompt,
            action_name=run_context.action_name,
            return_msg_type=msg_type,
            **kwargs,
        )
        run_context.execution_result = message
        self._record_trace(run_context, operation=MemoryOperation.EXECUTE, payload=message.content)

    async def _stage_reflect(self, run_context: MemoryRunContext, **kwargs: Any) -> None:
        policy = self._get_policy(MemoryOperation.REFLECT)
        if not policy.should_reflect(agent=self, run_context=run_context):
            logger.debug("[CustomizeMemoryAgent] Reflection skipped by policy.")
            return

        backends = policy.select_backends(
            operation=MemoryOperation.REFLECT,
            agent=self,
            run_context=run_context,
        )

        reflections = {}
        execution_message = run_context.execution_result
        if execution_message is None:
            logger.debug("[CustomizeMemoryAgent] Reflection skipped (no execution result).")
            return

        for backend in backends:
            try:
                reflection = await backend.dispatch(
                    MemoryOperation.REFLECT,
                    agent=self,
                    run_context=run_context,
                    execution_result=execution_message,
                    **kwargs,
                )
                if reflection is not None:
                    reflections[backend.name] = reflection
            except Exception as exc:
                logger.exception(
                    "[CustomizeMemoryAgent] Backend '%s' reflection failed: %s",
                    backend.name,
                    exc,
                )

        run_context.reflection_outputs = reflections
        self._record_trace(run_context, operation=MemoryOperation.REFLECT, payload=reflections)

    async def _stage_persist(self, run_context: MemoryRunContext, **kwargs: Any) -> None:
        if not run_context.reflection_outputs:
            logger.debug("[CustomizeMemoryAgent] Persist skipped (no reflection outputs).")
            return

        policy = self._get_policy(MemoryOperation.PERSIST)
        backends = policy.select_backends(
            operation=MemoryOperation.PERSIST,
            agent=self,
            run_context=run_context,
        )

        persist_results = {}
        for backend in backends:
            reflection = run_context.reflection_outputs.get(backend.name)
            if reflection is None:
                continue
            try:
                outcome = await backend.dispatch(
                    MemoryOperation.PERSIST,
                    agent=self,
                    run_context=run_context,
                    reflection=reflection,
                    **kwargs,
                )
                persist_results[backend.name] = outcome
            except Exception as exc:
                logger.exception(
                    "[CustomizeMemoryAgent] Backend '%s' persist failed: %s",
                    backend.name,
                    exc,
                )

        run_context.persistence_results = persist_results
        self._record_trace(run_context, operation=MemoryOperation.PERSIST, payload=persist_results)

    # ----------------------------------------------------------------- hooks --
    async def before_run(self, run_context: MemoryRunContext, **kwargs: Any) -> None:
        """Hook invoked prior to running the lifecycle pipeline."""
        del kwargs  # placeholder
        self.current_context = {
            "action_name": run_context.action_name,
            "inputs": run_context.inputs,
            "messages": run_context.messages,
        }
        self.execution_trace = []

    async def after_run(self, run_context: MemoryRunContext, **kwargs: Any) -> None:
        """Hook invoked once the lifecycle pipeline completes."""
        del kwargs  # placeholder
        self.current_context.update(
            {
                "result": run_context.execution_result,
                "reflections": run_context.reflection_outputs,
                "persisted": run_context.persistence_results,
            }
        )

    async def retrieve(self, query: Any, top_k: int = 3, **kwargs: Any) -> Dict[str, Any]:
        """Public method to trigger retrieval outside the standard pipeline."""
        metadata = kwargs.pop("metadata", None)
        run_context = MemoryRunContext(
            action_name="manual-retrieve",
            action=self.get_action(self.customize_action_name),
            inputs={},
            query=query,
            top_k=top_k,
        )
        if metadata:
            run_context.metadata.update(metadata)
        await self._stage_retrieve(run_context, **kwargs)
        return run_context.retrieved

    async def inject_context(self, inputs: dict, **kwargs: Any) -> dict:
        """Inject memory context into provided inputs using configured policies."""
        metadata = kwargs.pop("metadata", None)
        run_context = MemoryRunContext(
            action_name="manual-inject",
            action=self.get_action(self.customize_action_name),
            inputs=inputs,
        )
        run_context.retrieved = kwargs.pop("retrieved", {})
        if metadata:
            run_context.metadata.update(metadata)
        await self._stage_inject(run_context, **kwargs)
        return run_context.inputs

    async def execute_action(
        self,
        inputs: dict,
        action_name: Optional[str] = None,
        return_msg_type: MessageType = MessageType.UNKNOWN,
        **kwargs: Any,
    ) -> Message:
        """Execute only the action stage with the provided inputs."""
        metadata = kwargs.pop("metadata", None)
        target_action = self.get_action(action_name or self.customize_action_name)
        run_context = MemoryRunContext(
            action_name=target_action.name,
            action=target_action,
            inputs=inputs,
        )
        if metadata:
            run_context.metadata.update(metadata)
        await self._stage_execute(
            run_context=run_context,
            return_msg_type=return_msg_type,
            is_async=True,
            **kwargs,
        )
        return run_context.execution_result

    async def reflect_and_learn(self, result: Message, **kwargs: Any) -> Dict[str, Any]:
        """Trigger reflection (without persisting) for a provided execution result."""
        metadata = kwargs.pop("metadata", None)
        run_context = MemoryRunContext(
            action_name="manual-reflect",
            action=self.get_action(self.customize_action_name),
            inputs={},
        )
        run_context.execution_result = result
        if metadata:
            run_context.metadata.update(metadata)
        await self._stage_reflect(run_context, **kwargs)
        return run_context.reflection_outputs

    async def persist(self, **kwargs: Any) -> Dict[str, Any]:
        """Persist the latest reflection artifacts (if any)."""
        metadata = kwargs.pop("metadata", None)
        run_context = MemoryRunContext(
            action_name="manual-persist",
            action=self.get_action(self.customize_action_name),
            inputs={},
        )
        run_context.reflection_outputs = kwargs.pop("reflection_outputs", {})
        if metadata:
            run_context.metadata.update(metadata)
        await self._stage_persist(run_context, **kwargs)
        return run_context.persistence_results

    # -------------------------------------------------------------- utilities --
    def _record_trace(self, run_context: MemoryRunContext, operation: MemoryOperation, payload: Any) -> None:
        self.execution_trace.append({"operation": operation.value, "payload": payload})
        if self.memory_orchestrator:
            try:
                self.memory_orchestrator.after_stage(
                    agent=self,
                    run_context=run_context,
                    operation=operation,
                    result=payload,
                )
            except Exception:
                logger.exception("Memory orchestrator after_stage hook failed for %s", operation)

    def _get_policy(self, operation: MemoryOperation) -> BaseMemoryPolicy:
        policy = self.memory_policies.get(operation.value)
        if policy:
            return policy
        return self._default_policy

    def _ensure_backend_instance(
        self,
        name: str,
        backend: Any,
    ) -> BaseMemoryBackend:
        if isinstance(backend, BaseMemoryBackend):
            backend.name = backend.name or name
            return backend
        if isinstance(backend, dict):
            backend_instance = self._create_module_from_spec(backend, expected=BaseMemoryBackend)
            backend_instance.name = backend_instance.name or name
            return backend_instance
        if isinstance(backend, str):
            backend_cls = MODULE_REGISTRY.get_module(backend)
            backend_instance = backend_cls()
            backend_instance.name = backend_instance.name or name
            return backend_instance
        raise TypeError(f"Unsupported backend specification for '{name}': {type(backend)}")

    def _ensure_policy_instance(self, policy: Any) -> BaseMemoryPolicy:
        if isinstance(policy, BaseMemoryPolicy):
            return policy
        if isinstance(policy, dict):
            return self._create_module_from_spec(policy, expected=BaseMemoryPolicy)
        if isinstance(policy, str):
            policy_cls = MODULE_REGISTRY.get_module(policy)
            return policy_cls()
        raise TypeError(f"Unsupported policy specification: {type(policy)}")

    def _ensure_orchestrator_instance(self, orchestrator: Any) -> MemoryOrchestrator:
        if isinstance(orchestrator, MemoryOrchestrator):
            return orchestrator
        if isinstance(orchestrator, dict):
            return self._create_module_from_spec(orchestrator, expected=MemoryOrchestrator)
        if isinstance(orchestrator, str):
            orchestrator_cls = MODULE_REGISTRY.get_module(orchestrator)
            return orchestrator_cls()
        raise TypeError(f"Unsupported orchestrator specification: {type(orchestrator)}")

    def _create_module_from_spec(self, spec: Dict[str, Any], expected: Type[BaseModule]) -> BaseModule:
        if "class_name" not in spec:
            raise ValueError(f"Module spec missing 'class_name': {spec}")
        cls = MODULE_REGISTRY.get_module(spec["class_name"])
        module = cls._create_instance(spec)
        if not isinstance(module, expected):
            raise TypeError(f"Module '{spec['class_name']}' is not of type {expected.__name__}")
        return module
