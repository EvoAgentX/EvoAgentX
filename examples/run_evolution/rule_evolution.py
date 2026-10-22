"""V3: executable rule-program search, orchestrated by EvoAgentX. No LLM API calls."""
import os
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP','True')
import argparse
import json
import hashlib
import math
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from demo import TRAIN, VALIDATION, TEST, POLICY, selection, save_json

INITIAL_RULES={'refund_inclusive':False,'quality_required':False,
               'missing_escalates':False,'order_required':False}


def validate_input(ticket, rules):
    """Reject malformed input instead of silently coercing business facts."""
    if not isinstance(ticket, dict) or not isinstance(ticket.get('message'), str):
        raise ValueError('ticket.message must be a string')
    facts = ticket.get('facts')
    if not isinstance(facts, dict):
        raise ValueError('ticket.facts must be an object')
    if not isinstance(rules, dict) or set(rules) != set(INITIAL_RULES):
        raise ValueError('rules must contain exactly the four supported parameters')
    if any(type(value) is not bool for value in rules.values()):
        raise ValueError('rule parameters must be booleans')
    if 'days' in facts:
        days = facts['days']
        if type(days) not in (int, float) or not math.isfinite(days):
            raise ValueError('facts.days must be a finite number')
    for key in ('opened', 'defective'):
        if key in facts and type(facts[key]) is not bool:
            raise ValueError(f'facts.{key} must be a boolean')
    if 'order_id' in facts and not isinstance(facts['order_id'], str):
        raise ValueError('facts.order_id must be a string')


def classify(ticket):
    m=ticket['message']
    refund=any(w in m for w in ('退款','拿回钱'))
    exchange='换' in m
    shipping=any(w in m for w in ('物流','快递','订单到哪'))
    present=[c for c,ok in [('refund',refund),('exchange',exchange),('shipping',shipping)] if ok]
    return present[0] if len(present)==1 else 'other'


def decide(category,facts,rules):
    """Actual configurable decision program; never reads expected labels."""
    if category=='other': return {'category':category,'decision':'escalate','policy_id':'R4'}
    if category=='shipping':
        decision='request_info' if rules['order_required'] and not facts.get('order_id') else 'approve'
        return {'category':category,'decision':decision,'policy_id':'R3'}
    required=('days','opened') if category=='refund' else ('days','defective')
    if rules['missing_escalates'] and any(k not in facts for k in required):
        return {'category':category,'decision':'escalate','policy_id':'R4'}
    days=facts.get('days',0)
    if category=='refund':
        allowed=0<=days and (days<=7 if rules['refund_inclusive'] else days<7) and not facts.get('opened',False)
        rule='R1'
    else:
        allowed=0<=days<=30 and (not rules['quality_required'] or facts.get('defective',True))
        rule='R2'
    return {'category':category,'decision':'approve' if allowed else 'reject','policy_id':rule}


def validate_decision(answer):
    if set(answer)!= {'category','decision','policy_id'}: raise ValueError('Unexpected fields')
    if answer['category'] not in ('refund','exchange','shipping','other'): raise ValueError('category')
    if answer['decision'] not in ('approve','reject','request_info','escalate'): raise ValueError('decision')
    if answer['policy_id'] not in ('R1','R2','R3','R4'): raise ValueError('policy_id')
    return answer


from functools import cache

@cache
def action_classes():
    from evoagentx.workflow.action_graph import ActionGraph
    class Intake(ActionGraph):
        def execute(self,ticket: str)->dict:
            return {'category':classify(json.loads(ticket))}
    class Decision(ActionGraph):
        def execute(self,ticket: str,category: str,rules: str)->dict:
            answer=decide(category,json.loads(ticket)['facts'],json.loads(rules))
            return {'decision_json':json.dumps(answer)}
    class Audit(ActionGraph):
        def execute(self,decision_json: str)->dict:
            answer=validate_decision(json.loads(decision_json))
            return {'answer':json.dumps(answer)}
    return Intake,Decision,Audit


class Executor:
    def __init__(self):
        from evoagentx.models import LiteLLM,LiteLLMConfig
        from evoagentx.core.logging import logger
        logger.remove()
        self.config=LiteLLMConfig(model='openai/glm-4-flash',api_key='unused-no-network',
            api_base='https://open.bigmodel.cn/api/paas/v4/',is_local=True,stream=False)
        # Required by framework scheduler; ActionGraph tasks never invoke this model.
        self.llm=LiteLLM(config=self.config)
        self.executions=0

    def run(self,ticket,rules):
        validate_input(ticket, rules)
        from evoagentx.workflow import WorkFlow,WorkFlowGraph
        from evoagentx.workflow.workflow_graph import WorkFlowNode,WorkFlowEdge
        Intake,Decision,Audit=action_classes()
        def p(n): return {'name':n,'type':'string','required':True,'description':n}
        spec=[('intake',Intake,['ticket'],'category'),
              ('decision',Decision,['ticket','category','rules'],'decision_json'),
              ('audit',Audit,['decision_json'],'answer')]
        nodes=[WorkFlowNode(name=name,description=name,inputs=[p(n) for n in ins],outputs=[p(out)],
            action_graph=cls(name=name,description=name,llm_config=self.config)) for name,cls,ins,out in spec]
        graph=WorkFlowGraph(goal='执行可审计的售后决策程序',nodes=nodes,
            edges=[WorkFlowEdge(source='intake',target='decision'),WorkFlowEdge(source='decision',target='audit')],
            workflow_inputs=[p('ticket'),p('rules')],workflow_outputs=[p('answer')])
        result=WorkFlow(graph=graph,llm=self.llm).execute(
            {'ticket':json.dumps(ticket,ensure_ascii=False),'rules':json.dumps(rules)},extract_output=False)
        if result.status!='success': raise RuntimeError(result.error_msg)
        self.executions+=1
        return json.loads(result.result['answer'])


def evaluate(executor,cases,rules):
    if not cases:
        raise ValueError("Evaluation requires at least one case")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique")
    rows=[]
    for c in cases:
        answer=executor.run(c['input'],rules)
        rows.append({'id':c['id'],'group':c['group'],'input':c['input'],
                     'expected':c['expected'],'parsed':answer,'correct':answer==c['expected']})
    return {'accuracy':sum(r['correct'] for r in rows)/len(rows),'rows':rows}


def experiment(out):
    start=time.monotonic(); ex=Executor();out.mkdir(parents=True,exist_ok=True)
    best=dict(INITIAL_RULES)
    original=evaluate(ex,VALIDATION,best);current_validation=original
    history=[];all_trials=[]
    # Train-set hill climbing: one bit mutation per candidate. No oracle/reference config.
    for generation in range(1,5):
        train=evaluate(ex,TRAIN,best)
        trials=[]
        for key in sorted(best):
            candidate={**best,key:not best[key]}
            score=evaluate(ex,TRAIN,candidate)
            trials.append({'mutation':key,'rules':candidate,'train_accuracy':score['accuracy'],'train':score})
        all_trials.append({'generation':generation,'current_rules':best.copy(),'trials':trials})
        candidate=max(trials,key=lambda x:x['train_accuracy'])
        if candidate['train_accuracy']<=train['accuracy']: break
        validation=evaluate(ex,VALIDATION,candidate['rules'])
        accepted,fixed,regressed=selection(current_validation,validation)
        history.append({'generation':generation,'mutation':candidate['mutation'],
            'before_rules':best.copy(),'candidate_rules':candidate['rules'],
            'train_before':train['accuracy'],'train_after':candidate['train_accuracy'],
            'validation_before':current_validation['accuracy'],'validation_after':validation['accuracy'],
            'accepted':accepted,'fixed_validation_ids':fixed,'regressed_validation_ids':regressed})
        if not accepted: break
        best=candidate['rules'];current_validation=validation
        print(f'第{generation}代：{candidate["mutation"]}，验证准确率 {validation["accuracy"]:.1%}',flush=True)
    # Test labels never enter mutation, ranking, or selection above.
    initial=evaluate(ex,TEST,INITIAL_RULES);final=evaluate(ex,TEST,best)
    summary={'version':3,'mode':'deterministic_evoagentx_execution','seeded_defects':True,
        'optimizer':'one-bit mutation + training-score hill climbing + validation guard',
        'initial_accuracy':initial['accuracy'],'final_accuracy':final['accuracy'],
        'gain_percentage_points':round(100*(final['accuracy']-initial['accuracy']),2),
        'test_count':len(TEST),'generations':len(history),'workflow_executions':ex.executions,
        'llm_api_calls':0,'elapsed_seconds':round(time.monotonic()-start,2),
        'initial_rules':INITIAL_RULES,'final_rules':best,
        'limitations':['确定性规则程序的真实执行，不是LLM提示词进化或LLM成绩',
                      '有意设置四个可修复缺陷；候选空间仅16种配置',
                      '词法分类仅覆盖教学语句，不是通用自然语言理解',
                      '近分布合成测试，不能证明真实业务泛化或多Agent优于单模型']}
    result={'summary':summary,'history':history,'initial_test':initial,'final_test':final,
            'policy':POLICY,'train':TRAIN,'validation':VALIDATION,'test':TEST,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(out/'result.json',result);save_json(out/'search_trials.json',all_trials)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print('结果：',out/'result.json')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify',action='store_true',help='拒绝任何LLM调用，并验证搜索结果')
    args=ap.parse_args()
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    out=Path(__file__).resolve().parent/'output'/('rule_search_'+stamp)
    from unittest.mock import patch
    from evoagentx.models import LiteLLM
    # Always fail closed on accidental model invocation, not just in tests.
    with patch.object(LiteLLM,'single_generate',side_effect=AssertionError('Unexpected LLM call')), \
         patch.object(LiteLLM,'single_generate_async',side_effect=AssertionError('Unexpected LLM call')):
        result=experiment(out)
    if args.verify:
        assert result['summary']['initial_accuracy']==4/12
        assert result['summary']['final_accuracy']==1
        assert len(result['history'])==4
        assert all(h['accepted'] and not h['regressed_validation_ids'] for h in result['history'])
        assert all(r['correct'] for r in result['final_test']['rows'])
        print('PASS: actual EvoAgentX execution, search, held-out test; zero LLM calls')


if __name__=='__main__': main()
