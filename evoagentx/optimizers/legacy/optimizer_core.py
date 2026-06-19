from __future__ import annotations
import abc
import inspect
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


# Regular expression to match indexing expressions like foo[0] or bar["key"]
_INDEX_RE = re.compile(r'^(.*?)\[(.*?)\]$')


# ─────────────────────────────────────────────────────────────
# 1.  Runtime field helpers
# ─────────────────────────────────────────────────────────────
class OptimizableField:
    """Expose a concrete runtime attribute via get/set."""
    def __init__(self,
                 name: str,
                 getter: Callable[[], Any],
                 setter: Callable[[Any], None]):
        self.name, self._get, self._set = name, getter, setter
    def get(self) -> Any:            return self._get()
    def set(self, value: Any) -> None: self._set(value)


class PromptRegistry:
    """Central registry for all runtime-patchable fields."""
    def __init__(self) -> None:
        self.fields: Dict[str, OptimizableField] = {}
    def register_field(self, field: OptimizableField):
        self.fields[field.name] = field
    # convenience
    def get(self, name: str) -> Any:
        return self.fields[name].get()
    def set(self, name: str, value: Any):
        self.fields[name].set(value)
    def names(self) -> List[str]:
        return list(self.fields.keys())

    # -- 新增 API ----------------------------------------------
    def register_path(self, root: Any, path: str, *, name: str|None=None):
        """用类似 'encoder.layers[3].dropout_p' 的字符串一次性注册。"""
        key = name or path.split(".")[-1]          # 建议让用户自起更短 alias
        parent, leaf = self._walk(root, path)

        def getter():                       # 读
            return parent[leaf] if isinstance(parent, (list, dict)) else getattr(parent, leaf)

        def setter(v):                      # 写
            if isinstance(parent, (list, dict)):
                parent[leaf] = v
            else:
                setattr(parent, leaf, v)

        field = OptimizableField(key, getter, setter)
        self.register_field(field)
        return field

    def _walk(self, root, path: str, create_missing=False):
        cur = root
        parts = path.split(".")
        for part in parts[:-1]:
            m = _INDEX_RE.match(part)
            if m:
                attr, idx = m.groups()
                cur = getattr(cur, attr) if attr else cur
                idx = idx.strip()
                if (idx.startswith("'") and idx.endswith("'")) or (idx.startswith('"') and idx.endswith('"')):
                    idx = idx[1:-1]  # strip quotes if it's a string key
                elif idx.isdigit():
                    idx = int(idx)
                cur = cur[idx]
            else:
                cur = getattr(cur, part)

        # 最后一个叶子属性
        leaf = parts[-1]
        m = _INDEX_RE.match(leaf)
        if m:
            attr, idx = m.groups()
            parent = getattr(cur, attr) if attr else cur
            idx = idx.strip()
            if (idx.startswith("'") and idx.endswith("'")) or (idx.startswith('"') and idx.endswith('"')):
                idx = idx[1:-1]
            elif idx.isdigit():
                idx = int(idx)
            return parent, idx
        return cur, leaf


# ─────────────────────────────────────────────────────────────
# 2.  CodeBlock  (sync / async dual‑compatible)
# ─────────────────────────────────────────────────────────────
# result = await block.run(cfg)    
class CodeBlock:
    """
    Parameters
    ----------
    name : str
        逻辑名（日志、调试友好）
    func : Callable[[dict], Any]
        普通同步函数，输入 cfg 字典
    """

    def __init__(self, name: str, func: Callable[[Dict[str, Any]], Any]):
        self.name = name
        self._func = func

    def run(self, cfg: Dict[str, Any]) -> Any:
        """同步执行封装的函数。"""
        return self._func(cfg)

    def __call__(self, cfg: Dict[str, Any]) -> Any:
        return self.run(cfg)

    def __repr__(self):
        return f"<CodeBlock {self.name} (sync)>"




# ─────────────────────────────────────────────────────────────
# 3.  BaseCodeBlockOptimizer
# ─────────────────────────────────────────────────────────────
class BaseCodeBlockOptimizer(abc.ABC):
    """
    Abstract optimiser that:
      • performs sequential trials
      • writes sampled cfg back to runtime via PromptRegistry
      • validates that registered names appear in CodeBlock signature
    """

    def __init__(self,
                 registry: PromptRegistry,
                 metric: str,
                 maximize: bool = True,
                 max_trials: int = 30):
        self.registry   = registry
        self.metric     = metric
        self.maximize   = maximize
        self.max_trials = max_trials

    @abc.abstractmethod
    def sample_cfg(self) -> Dict[str, Any]:
        """Return a cfg dict (may include subset of registry names)."""

    @abc.abstractmethod
    def update(self, cfg: Dict[str, Any], score: float):
        """Update internal optimiser state."""

    def _apply_cfg(self, cfg: Dict[str, Any]):
        for k, v in cfg.items():
            if k in self.registry.fields:
                self.registry.set(k, v)

    def _check_codeblock_compat(self, code_block: CodeBlock):
        sig = inspect.signature(code_block._func)
        params = sig.parameters.values()

        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        accepts_cfg_dict = "cfg" in sig.parameters

        if has_kwargs or accepts_cfg_dict:
            return

        allowed_keys = set(sig.parameters)
        unknown = set(self.registry.names()) - allowed_keys
        if unknown:
            import warnings
            warnings.warn(f"PromptRegistry fields {unknown} are not present in "
                          f"{code_block.name}() signature; they will be ignored.")

    def run(self,
            code_block: CodeBlock,
            evaluator: Callable[[Dict[str, Any], Any], float]
            ) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], float]]]:

        self._check_codeblock_compat(code_block)

        best_cfg, best_score = None, -float("inf") if self.maximize else float("inf")
        history: List[Tuple[Dict[str, Any], float]] = []

        for _ in range(self.max_trials):
            cfg = self.sample_cfg()
            self._apply_cfg(cfg)
            result = code_block.run(cfg)
            score = evaluator(cfg, result)
            self.update(cfg, score)

            history.append((cfg, score))
            better = score > best_score if self.maximize else score < best_score
            if better:
                best_cfg, best_score = cfg, score

        return best_cfg, history



# ────────────────────────────────────────────────────────────
# Other  Helper: bind_cfg – write cfg into nested attributes
# ────────────────────────────────────────────────────────────
def bind_cfg(obj: Any, cfg: Dict[str, Any]) -> None:
    """Recursively write *cfg* values into (potentially nested) attributes
    of *obj*.  Key like "a.b.c" becomes obj.a.b.c = value.
    """
    for key, val in cfg.items():
        parts = key.split(".")
        cur = obj
        for part in parts[:-1]:
            cur = getattr(cur, part)
        setattr(cur, parts[-1], val)



# Demo
# ───────────────────── ───────────────────── ──────────────────── #
# ───────────────────── ───────────────────── ──────────────────── #
# ───────────────────── Demo: 业务对象 & 工作流 ──────────────────── #
# ─────────────────────── Demo: Workflow & Sampler ─────────────────────── #
@dataclass
class Sampler:
    temperature: float = 0.7
    top_p: float = 0.9

class Workflow:

    def __init__(self):
        self.system_prompt = "You are a helpful assistant."
        self.few_shot = "Q: 1+1=?\nA: 2"
        self.sampler = Sampler()

    # @parameter_registry("name", ["a", "self.system_prompt"])
    def execute(self):
        # a = 000 
        pass 

    def run(self):
        prompt = f"{self.system_prompt}\n{self.few_shot}\nUser: Hi"
        return {"prompt": prompt, "score": random.uniform(0, 1)}


# ─────────────────────── Optimizer 实现 ─────────────────────── #
class RandomSearchOptimizer(BaseCodeBlockOptimizer):
    def sample_cfg(self) -> Dict[str, Any]:
        return {
            "sampler_temperature": random.uniform(0.3, 1.3),
            "sampler_top_p":       random.uniform(0.5, 1.0),
            "sys_prompt": random.choice([
                "You are a helpful assistant.",
                "You are a super-concise assistant."
            ]),
        }

    def update(self, cfg, score):
        pass


class GreedyLoggerOptimizer(BaseCodeBlockOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best = None
        self.best_score = -float("inf") if self.maximize else float("inf")

    def sample_cfg(self):
        return {
            "sampler_temperature": random.uniform(0.3, 1.3),
            "sampler_top_p":       random.uniform(0.5, 1.0),
            "sys_prompt": random.choice([
                "You are a helpful assistant.",
                "You are a super-concise assistant."
            ]),
        }

    def update(self, cfg, score):
        if (self.maximize and score > self.best_score) or (not self.maximize and score < self.best_score):
            self.best = cfg
            self.best_score = score
            print(f"[New Best] score={score:.3f} cfg={cfg}")



# ─────────────────────── 实验入口 ─────────────────────── #
def main():
    flow = Workflow()

    registry = PromptRegistry()
    registry.register_path(flow, "system_prompt", name="sys_prompt")
    registry.register_path(flow, "sampler.temperature")
    registry.register_path(flow, "sampler.top_p")

    code_block = CodeBlock("run_workflow", lambda cfg: flow.run())

    def evaluator(cfg, result) -> float:
        return result["score"]

    opt = RandomSearchOptimizer(registry, metric="score", max_trials=10)
    best_cfg, history = opt.run(code_block, evaluator)

    print("\n=== Trial history ===")
    for i, (cfg, score) in enumerate(history, 1):
        print(f"{i:02d}: score={score:.3f}, cfg={cfg}")

    print("\n=== Best ===")
    print(best_cfg)


if __name__ == "__main__":
    main()
