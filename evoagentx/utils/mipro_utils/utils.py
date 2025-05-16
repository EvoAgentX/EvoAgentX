# 复制黏贴的dspy/dspy/dsp/utils/utils.py
import os
import re
import numpy as np
import tqdm
import copy
import random
import json
import logging
import datetime
import itertools
from collections import defaultdict
from evoagentx.utils.mipro_utils.labeledfewshot import LabeledFewShot
from evoagentx.utils.mipro_utils.bootstrap import BootstrapFewShot

logger = logging.getLogger("MIPRO")

def print_message(*s, condition=True, pad=False, sep=None):
    s = " ".join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        msg = msg if not pad else f"\n{msg}\n"
        print(msg, flush=True, sep=sep)

    return msg


def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def file_tqdm(file):
    print(f"#> Reading {file.name}")

    with tqdm.tqdm(
        total=os.path.getsize(file.name) / 1024.0 / 1024.0, unit="MiB",
    ) as pbar:
        for line in file:
            yield line
            pbar.update(len(line) / 1024.0 / 1024.0)

        pbar.close()


def create_directory(path):
    if os.path.exists(path):
        print("\n")
        print_message("#> Note: Output directory", path, "already exists\n\n")
    else:
        print("\n")
        print_message("#> Creating directory", path, "\n\n")
        os.makedirs(path)


def deduplicate(seq: list[str]) -> list[str]:
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset : offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


# class dotdict(dict):
#     """
#     dot.notation access to dictionary attributes
#     Credit: derek73 @ https://stackoverflow.com/questions/2352181
#     """

#     __getattr__ = dict.__getitem__
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


class dotdict(dict):
    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            return super().__getattr__(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __delattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            super().__delattr__(key)
        else:
            del self[key]

    def __deepcopy__(self, memo):
        # Use the default dict copying method to avoid infinite recursion.
        return dotdict(copy.deepcopy(dict(self), memo))


class dotdict_lax(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(L):
    # return [x for y in L for x in y]

    result = []
    for _list in L:
        result += _list

    return result


def zipstar(L, lazy=False):
    """
    A much faster A, B, C = zip(*[(a, b, c), (a, b, c), ...])
    May return lists or tuples.
    """

    if len(L) == 0:
        return L

    width = len(L[0])

    if width < 100:
        return [[elem[idx] for elem in L] for idx in range(width)]

    L = zip(*L)

    return L if lazy else list(L)


def zip_first(L1, L2):
    length = len(L1) if type(L1) in [tuple, list] else None

    L3 = list(zip(L1, L2))

    assert length in [None, len(L3)], "zip_first() failure: length differs!"

    return L3


def int_or_float(val):
    if "." in val:
        return float(val)

    return int(val)


def groupby_first_item(lst):
    groups = defaultdict(list)

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest
        groups[first].append(rest)

    return groups


def process_grouped_by_first_item(lst):
    """
    Requires items in list to already be grouped by first item.
    """

    groups = defaultdict(list)

    started = False
    last_group = None

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest

        if started and first != last_group:
            yield (last_group, groups[last_group])
            assert (
                first not in groups
            ), f"{first} seen earlier --- violates precondition."

        groups[first].append(rest)

        last_group = first
        started = True

    return groups


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def lengths2offsets(lengths):
    offset = 0

    for length in lengths:
        yield (offset, offset + length)
        offset += length

    return


# see https://stackoverflow.com/a/45187287
class NullContextManager:
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


def load_batch_backgrounds(args, qids):
    if args.qid2backgrounds is None:
        return None

    qbackgrounds = []

    for qid in qids:
        back = args.qid2backgrounds[qid]

        if len(back) and isinstance(back[0], int):
            x = [args.collection[pid] for pid in back]
        else:
            x = [args.collectionX.get(pid, "") for pid in back]

        x = " [SEP] ".join(x)
        qbackgrounds.append(x)

    return qbackgrounds

def create_minibatch(trainset, batch_size=50, rng=None):
    """Create a minibatch from the trainset."""

    # Ensure batch_size isn't larger than the size of the dataset
    batch_size = min(batch_size, len(trainset))

    # If no RNG is provided, fall back to the global random instance
    rng = rng or random

    # Randomly sample indices for the mini-batch using the provided rng
    sampled_indices = rng.sample(range(len(trainset)), batch_size)

    # Create the mini-batch using the sampled indices
    minibatch = [trainset[i] for i in sampled_indices]

    return minibatch

def create_n_fewshot_demo_sets(
    student,
    num_candidate_sets,
    trainset,
    collate_func,
    max_labeled_demos,
    max_bootstrapped_demos,
    metric,
    max_errors=10,
    max_rounds=1,
    labeled_sample=True,
    min_num_samples=1,
    metric_threshold=None,
    teacher=None,
    include_non_bootstrapped=True,
    seed=0,
    rng=None
):
    """
    This function is copied from random_search.py, and creates fewshot examples in the same way that random search does.
    This allows us to take advantage of using the same fewshot examples when we use the same random seed in our optimizers.
    """
    demo_candidates = {}

    # Account for confusing way this is set up, where we add in 3 more candidate sets to the N specified
    num_candidate_sets -= 3

    logger.info("Working with create_n_fewshot_demo_sets")
    # Initialize demo_candidates dictionary
    for agent in student.agents():
        demo_candidates[agent['name']] = []


    rng = rng or random.Random(seed)

    # Go through and create each candidate set
    for seed in range(-3, num_candidate_sets):

        print(f"Bootstrapping set {seed+4}/{num_candidate_sets+3}")

        trainset_copy = list(trainset)

        if seed == -3 and include_non_bootstrapped:
            # zero-shot
            program2 = student.reset_copy()
            logger.info("zero-shot finished")
        elif (
            seed == -2
            and max_labeled_demos > 0
            and include_non_bootstrapped
        ):
            # labels only
            program = LabeledFewShot(k=max_labeled_demos)
            program2 = program.optimize(
                student, trainset=trainset_copy, sample=labeled_sample,
            )
        elif seed == -1:
            # unshuffled few-shot
            program = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                max_rounds=max_rounds,

            )
            program2 = program.optimize(student, teacher=teacher, trainset=trainset_copy, collate_func=collate_func)
            logger.info("bootstrap few-shot finished")
        else:
            # shuffled few-shot
            rng.shuffle(trainset_copy)
            size = rng.randint(min_num_samples, max_bootstrapped_demos)

            program = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                max_rounds=max_rounds,
            )

            program2 = program.optimize(
                student, teacher=teacher, trainset=trainset_copy, collate_func=collate_func
            )

        for i, agent in enumerate(student.agents()):
            demo_candidates[agent['name']].append(program2.agents()[i]['demos'])
    return demo_candidates

def get_source_code(program):
    """
    Get the source code of a program.
    """
    return json.dumps(program.get_graph_info(), indent=2, ensure_ascii=False)

def strip_prefix(text):
    pattern = r'^[\*\s]*(([\w\'\-]+\s+){0,4}[\w\'\-]+):\s*'
    modified_text = re.sub(pattern, '', text)
    return modified_text.strip("\"")

def order_input_keys_in_string(unordered_repr):
    # Regex pattern to match the input keys structure
    pattern = r"input_keys=\{([^\}]+)\}"
    
    def reorder_keys(match):
        # Extracting the keys from the match
        keys_str = match.group(1)
        # Splitting the keys, stripping extra spaces, and sorting them
        keys = sorted(key.strip() for key in keys_str.split(','))
        # Formatting the sorted keys back into the expected structure
        return f"input_keys={{{', '.join(keys)}}}"

    # Using re.sub to find all matches of the pattern and replace them using the reorder_keys function
    ordered_repr = re.sub(pattern, reorder_keys, unordered_repr)

    return ordered_repr

def create_predictor_level_history_string(base_program, predictor_i, trial_logs, top_n):
    instruction_aggregate = {}
    instruction_history = []
    
    # Load trial programs
    for trial_num in trial_logs:
        trial = trial_logs[trial_num]
        if "program_path" in trial:
            trial_program = base_program.deepcopy()
            # must have a graph_path to separate the objects
            trial_program.from_file(trial.graph_path)
            instruction_history.append({
                "program": trial_program,
                "score": trial["score"],
            })

    # Aggregate scores for each instruction
    for history_item in instruction_history:
        agent = history_item["program"].agents()[predictor_i]
        # TODO: 后续添加关于system prompt的内容
        
        instruction = agent['prompt']
        score = history_item["score"]
        
        if instruction in instruction_aggregate:
            instruction_aggregate[instruction]['total_score'] += score
            instruction_aggregate[instruction]['count'] += 1
        else:
            instruction_aggregate[instruction] = {'total_score': score, 'count': 1}
    
    # Calculate average score for each instruction and prepare for sorting
    predictor_history = []
    for instruction, data in instruction_aggregate.items():
        average_score = data['total_score'] / data['count']
        predictor_history.append((instruction, average_score))
    
    # Deduplicate and sort by average score, then select top N
    seen_instructions = set()
    unique_predictor_history = []
    for instruction, score in predictor_history:
        if instruction not in seen_instructions:
            seen_instructions.add(instruction)
            unique_predictor_history.append((instruction, score))

    top_instructions = sorted(unique_predictor_history, key=lambda x: x[1], reverse=True)[:top_n]
    top_instructions.reverse()
    
    # Create formatted history string
    predictor_history_string = ""
    for instruction, score in top_instructions:
        predictor_history_string += instruction + f" | Score: {score}\n\n"
    
    return predictor_history_string

def create_example_string(fields, example):

    # Building the output string
    output = []
    for field_name, field_values in fields.items():
        name = field_values.json_schema_extra["prefix"]

        # Determine the value from input_data or prediction_data
        value = example.get(field_name)

        # Construct the string for the current field
        field_str = f"{name} {value}"
        output.append(field_str)

    # Joining all the field strings
    return '\n'.join(output)


def eval_candidate_program(batch_size, trainset, collate_func, candidate_program, evaluate, rng=None, return_all_scores=False):
    """Evaluate a candidate program on the trainset, using the specified batch size."""

    try:
        # Evaluate on the full trainset
        if batch_size >= len(trainset):
            return evaluate(candidate_program, devset=trainset, return_all_scores=return_all_scores, collate_func=collate_func)
        # Or evaluate on a minibatch
        else:
            return evaluate(
                candidate_program,
                devset=create_minibatch(trainset, batch_size, rng),
                return_all_scores=return_all_scores,
                collate_func=collate_func
            )
    except Exception:
        logger.error("An exception occurred during evaluation", exc_info=True)
        if return_all_scores:
            return 0.0, [0.0] * len(trainset)
        return 0.0  # TODO: Handle this better, as -ve scores are possible


def get_program_with_highest_avg_score(param_score_dict, fully_evaled_param_combos):
    """Used as a helper function for bayesian + minibatching optimizers. Returns the program with the highest average score from the batches evaluated so far."""

    # Calculate the mean for each combination of categorical parameters, based on past trials
    results = []
    for key, values in param_score_dict.items():
        scores = np.array([v[0] for v in values])
        mean = np.average(scores)
        program = values[0][1]
        params = values[0][2]
        results.append((key, mean, program, params))

    # Sort results by the mean
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Find the combination with the highest mean, skip fully evaluated ones
    for combination in sorted_results:
        key, mean, program, params = combination

        if key in fully_evaled_param_combos:
            continue

        return program, mean, key, params

    # If no valid program is found, we return the last valid one that we found
    return program, mean, key, params

def save_candidate_program(program, save_path, trial_num, note=None):
    """Save the candidate program to the log directory.
    
    Args:
        program: The program to save
        save_path: Directory to save the program in, can be a full path including .json
        trial_num: Trial number for the program
        note: Optional note to add to the filename
        
    Returns:
        str: Path where the program was saved, or None if save_path is None
    """
    if save_path is None:
        return None

    # 只保留目录部分
    save_dir = os.path.dirname(save_path)

    # 确保目录存在
    eval_programs_dir = os.path.join(save_dir, "evaluated_programs")
    os.makedirs(eval_programs_dir, exist_ok=True)

    # 定义保存路径
    if note:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}_{note}.json")
    else:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}.json")

    # 保存程序
    program.save_module(save_path)

    return save_path