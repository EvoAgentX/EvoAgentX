from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import math

from ...models.model_configs import LLMConfig
from ...agents.customize_agent import CustomizeAgent


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.strip()]


def _tf_vector(tokens: List[str]) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    # l2 normalize
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    for k in list(vec.keys()):
        vec[k] /= norm
    return vec


def _cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) < len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    # simple smoothed unigram distributions over union vocab
    vocab = set(p.keys()) | set(q.keys())
    eps = 1e-9
    def _norm(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.get(w, 0.0) for w in vocab) or 1.0
        return {w: (d.get(w, 0.0) + eps) / (s + eps * len(vocab)) for w in vocab}
    P = _norm(p)
    Q = _norm(q)
    M = {w: 0.5 * (P[w] + Q[w]) for w in vocab}
    def _kl(X, Y):
        return sum(X[w] * math.log((X[w] + eps) / (Y[w] + eps)) for w in vocab)
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)


class PruningPipeline:
    """可插拔剪枝流水线：质量剪枝(QP) → 多样性剪枝(DP) → 误解反驳(MR)。

    候选输入格式：List[{"agent_id": int, "text": str}]
    输出保留相同结构，并在条目中填充可选指标：qp_score、dup_removed 等。
    """

    def __init__(
        self,
        enable_qp: bool = True,
        enable_dp: bool = True,
        enable_mr: bool = False,
        qp_threshold: float = 0.15,
        qp_top_k: Optional[int] = None,
        dp_similarity_threshold: float = 0.92,
        dp_max_candidates: Optional[int] = None,
        mr_llm_config: Optional[LLMConfig] = None,
        min_keep_count: Optional[int] = None,
    ) -> None:
        self.enable_qp = enable_qp
        self.enable_dp = enable_dp
        self.enable_mr = enable_mr
        self.qp_threshold = qp_threshold
        self.qp_top_k = qp_top_k
        self.dp_similarity_threshold = dp_similarity_threshold
        self.dp_max_candidates = dp_max_candidates
        self.mr_llm_config = mr_llm_config
        # 最少保留条数（基于参与人数由上层计算传入）。若为 None，则不强制最小保留。
        self.min_keep_count = min_keep_count

    # -------------------- QP: 质量剪枝 --------------------
    def _qp_score(self, problem: str, text: str) -> float:
        # 简易相关性评分：基于词袋余弦
        qv = _tf_vector(_tokenize(problem))
        tv = _tf_vector(_tokenize(text))
        return _cosine_sim(qv, tv)

    def _quality_prune(self, problem: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enable_qp or len(candidates) <= 1:
            return candidates
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for c in candidates:
            s = self._qp_score(problem, c.get("text", ""))
            c = dict(c)
            c["qp_score"] = s
            scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        if self.qp_top_k is not None and self.qp_top_k > 0:
            scored = scored[: self.qp_top_k]
        kept = [c for s, c in scored if s >= self.qp_threshold]
        # 至少保留一个；若配置了最少保留，补齐到该数量
        if not kept:
            kept = [scored[0][1]]
        if self.min_keep_count and len(kept) < self.min_keep_count:
            # 依据分数从高到低补齐
            existing_ids = set(id(obj) for obj in kept)
            for _, c in scored:
                if id(c) not in existing_ids:
                    kept.append(c)
                if len(kept) >= self.min_keep_count:
                    break
        return kept

    # -------------------- DP: 多样性剪枝 --------------------
    def _diversity_prune(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enable_dp or len(candidates) <= 1:
            return candidates
        vecs = [_tf_vector(_tokenize(c.get("text", ""))) for c in candidates]
        kept: List[int] = []
        for i, v in enumerate(vecs):
            diverse = True
            for j in kept:
                sim = _cosine_sim(v, vecs[j])
                if sim >= self.dp_similarity_threshold:
                    diverse = False
                    break
            if diverse:
                kept.append(i)
            if self.dp_max_candidates and len(kept) >= self.dp_max_candidates:
                break
        pruned = [candidates[i] for i in kept]
        # 若设置了最少保留，尝试补齐（优先根据 qp_score 从高到低补）
        if self.min_keep_count and len(pruned) < self.min_keep_count:
            # 构造按 qp_score 的全局顺序
            ranked = sorted(
                range(len(candidates)),
                key=lambda idx: float(candidates[idx].get("qp_score") or 0.0),
                reverse=True,
            )
            chosen = set(kept)
            for idx in ranked:
                if idx in chosen:
                    continue
                pruned.append(candidates[idx])
                chosen.add(idx)
                if len(pruned) >= self.min_keep_count:
                    break
        return pruned

    # -------------------- MR: 误解反驳 --------------------
    def _build_critic(self) -> Optional[CustomizeAgent]:
        if not self.mr_llm_config:
            return None
        prompt = (
            """
You are a critical reviewer. Given a problem and a set of condensed candidate answers, identify common misunderstandings or mistakes, and propose a corrected consolidated answer.

Problem:
{problem}

Candidates:
{candidates_text}

Return XML:
<response>
  <issues>Common mistakes found</issues>
  <rebuttal>How to fix them</rebuttal>
  <corrected>Single corrected final answer</corrected>
</response>
            """
        ).strip()
        inputs = [
            {"name": "problem", "type": "str", "description": "Problem statement"},
            {"name": "candidates_text", "type": "str", "description": "Concatenated candidates"},
        ]
        outputs = [
            {"name": "issues", "type": "str", "description": "Common mistakes", "required": True},
            {"name": "rebuttal", "type": "str", "description": "Corrections", "required": True},
            {"name": "corrected", "type": "str", "description": "Corrected final answer", "required": True},
        ]
        return CustomizeAgent(
            name="CriticAgent",
            description="Detects misunderstandings and proposes corrected answer",
            prompt=prompt,
            llm_config=self.mr_llm_config,
            inputs=inputs,
            outputs=outputs,
            parse_mode="xml",
        )

    def _misunderstanding_rebuttal(self, problem: str, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, str]]]:
        if not self.enable_mr:
            return candidates, None
        critic = self._build_critic()
        if critic is None:
            return candidates, None
        concat = "\n\n".join(f"#{c.get('agent_id')}: {c.get('text','').strip()}" for c in candidates)
        msg = critic(inputs={"problem": problem, "candidates_text": concat})
        st = msg.content.get_structured_data()
        # 将批判结果附加到候选（不改变原文），并返回一个建议的最终答案
        for c in candidates:
            c["mr_issues"] = st.get("issues", "")
            c["mr_rebuttal"] = st.get("rebuttal", "")
        suggested = {
            "issues": st.get("issues", ""),
            "rebuttal": st.get("rebuttal", ""),
            "corrected": st.get("corrected", ""),
        }
        return candidates, suggested

    # -------------------- Pipeline entry --------------------
    def apply(self, problem: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """返回 {"candidates": pruned, "mr_suggested": optional}。"""
        step1 = self._quality_prune(problem, candidates)
        step2 = self._diversity_prune(step1)
        step3, suggested = self._misunderstanding_rebuttal(problem, step2)
        return {"candidates": step3, "mr_suggested": suggested}


