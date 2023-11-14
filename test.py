from multiprocessing import Pool
from time import sleep

import copy
import os
import openai
import rich
from tqdm import tqdm
import torch
import random
import json
import nltk
import argparse
from datetime import datetime
from models.model import ChatGPTModel, ChatModel, GlmChatModel, RandomModel, models
from utils.fn import lcs
from utils.common import option_label
from utils.logger import Logger
from rich.prompt import Confirm
from string import punctuation

from utils.metric import get_possible_answers

punctuation = set(punctuation) | {'``', "''", '“', '”', '‘', '’', ','}


class RequestWarpper:

    def __init__(self,
                 model,
                 exemplars,
                 seed,
                 exemplar_type,
                 n_exemplar,
                 dry_run=False):
        self.seed = seed
        self.model = model
        self.exemplars = exemplars
        self.exemplar_type = exemplar_type
        self.n_exemplar = n_exemplar
        self.dry_run = dry_run

    def __call__(self, instance):
        model = self.model
        question_id = instance["id"]
        question_type = instance['type']

        # Set random seed, according to question_id and seed
        # Ensure the same exemplars are used for the same question for all models
        random.seed(f"{question_id} {self.seed}")
        if self.n_exemplar > 0:
            if self.exemplar_type == "generation-method":
                question_generation_method = instance[
                    'generation_method'].replace("_not", "")
                exemplars = random.sample(
                    self.
                    exemplars[f"{question_generation_method}@{question_type}"],
                    self.n_exemplar)
            elif self.exemplar_type == "syntactic-knowledge":
                question_syntactic_knowledge_point = instance[
                    'knowledge_point']
                exemplars = random.sample(
                    self.exemplars[
                        f"{question_syntactic_knowledge_point}@{question_type}"],
                    self.n_exemplar)
            elif self.exemplar_type == "exclude-self-syntactic-knowledge":
                question_syntactic_knowledge_point = instance[
                    'knowledge_point']
                exemplars = random.sample(
                    sum([
                        self.exemplars[t] for t in self.exemplars.keys()
                        if (question_type in t
                            and question_syntactic_knowledge_point not in t)
                    ], []), self.n_exemplar)
            elif self.exemplar_type == "all":
                exemplars = random.sample(self.exemplars[question_type],
                                          self.n_exemplar)
            else:
                raise NotImplementedError(
                    f"Exemplar type {self.exemplar_type} not implemented")
        else:
            exemplars = []

        request = self.model.build_request(exemplars, instance)
        instance["request"] = request

        n_tries = 0
        if not self.dry_run:
            while True:
                try:
                    answers = model.get_answer(request, question_type)
                    break
                except (openai.error.RateLimitError,
                        openai.error.ServiceUnavailableError,
                        openai.error.APIError):
                    rich.print(
                        f"Evaluating question ({question_id}), the {n_tries + 1}th try...",
                        end="\r")
                    sleep(0.1)
                    n_tries += 1
        else:
            answers = ["Answer: <Fake answer>"]

        return instance, answers


def tokenize(text):
    # preprocess
    text = text.lower()
    text = text.replace("\/", "/")
    words = nltk.word_tokenize(text)
    # remove punct
    return [word for word in words if word not in punctuation]


def dump_predictions(predictions, path):
    json.dump(predictions, open(path, "w"), indent=4)


def main(args):

    model_name = args.model_name
    if model_name == "random" or args.dry_run:
        args.save_results_per = 10000000000

    if args.n_workers is None:
        model_class = models.get(model_name, models["default"])
        if model_class == ChatGPTModel:
            args.n_workers = 4
        elif model_name == "random" or args.dry_run:
            args.n_workers = 8
        else:
            args.n_workers = 1

    # mkdir -p data/model_name
    n_shots = f"{args.n_exemplar}-shot"
    if args.fine_grained_exemplars:
        rich.print(
            "[bold orange_red1]Warning[/bold orange_red1]: This option has been deprecated, please use --exemplar-type instead"
        )
        args.exemplar_type = "generation-method"

    sample_size = args.sample_size

    if args.n_exemplar > 0:
        if args.exemplar_type == "generation-method":
            n_shots = f"{n_shots}-fine-grained"
        elif args.exemplar_type == "syntactic-knowledge":
            n_shots = f"{n_shots}-syntactic"
        elif args.exemplar_type == "exclude-self-syntactic-knowledge":
            n_shots = f"{n_shots}-exclude-self-syntactic"

    model_name_str = model_name
    if args.model_revision != 'main':
        model_name_str = f"{model_name_str}-{args.model_revision}"
    if args.lora_weights is not None:
        model_name_str = f"{model_name_str}-lora-finetuned"

    if args.task_name == "mmlu":
        args.suite = f"mmlu.{args.suite}"
    if args.max_eval_instances is not None:
        exp_path = f"exp/{model_name_str}/{args.suite}-{args.max_eval_instances}/{sample_size}-sampled-test-set/{n_shots}/seed-{args.seed}"
    else:
        exp_path = f"exp/{model_name_str}/{args.suite}/{sample_size}-sampled-test-set/{n_shots}/seed-{args.seed}"

    os.makedirs(exp_path, exist_ok=True)
    logger = Logger(f"{exp_path}/results.txt")

    if args.task_name == "mmlu":
        assert args.exemplar_type == "subject"
        args.exemplar_type = "syntactic-knowledge"
        exemplars = json.load(open(f"data/mmlu/exemplars.json", "r"))
        test_set = json.load(open(f"data/mmlu/test.json", "r"))
        explain_test_set = None
    else:
        if args.exemplar_type == "generation-method":
            exemplars = json.load(
                open("generated/test.fine-grained-exemplars.json", "r"))
        elif args.exemplar_type in {
                "syntactic-knowledge", "exclude-self-syntactic-knowledge"
        }:
            exemplars = json.load(
                open("generated/test.syntactic-exemplars.json", "r"))
        elif args.exemplar_type == "all":
            exemplars = json.load(open("generated/test.exemplars.json", "r"))
        else:
            raise NotImplementedError(
                f"Exemplar type {args.exemplar_type} not implemented")

        test_set = json.load(
            open(f"generated/test.random-sample-{sample_size}.json", "r"))
        if args.explain_syntactic_knowledge and args.exemplar_type == "syntactic-knowledge":
            explain_test_set = json.load(
                open("generated/test.syntactic-knowledge.json", "r"))
        else:
            explain_test_set = None

    # dump code version, and description
    # if exists check if is the same as the current code
    # if not force re-evaluation, get user a warning, and ask for confirmation
    if not args.force and not args.ignore_code_revision and os.path.exists(
            f"{exp_path}/code_version.txt"):
        with open(f"{exp_path}/code_version.txt", "r") as f:
            code_version = f.read().strip()
        try:
            with open(f"{exp_path}/code_diff.txt", "r") as f:
                code_diff = f.read().strip()
        except FileNotFoundError:
            code_diff = ""
        current_code_version = os.popen("git rev-parse HEAD").read().strip()
        if code_version != current_code_version:
            rich.print(
                f"Old code version: {code_version}, current code version: {current_code_version}"
            )
            if not Confirm.ask(
                    "[bold orange_red1]Warning[/bold orange_red1]: Code version mismatch, please confirm if you want to continue"
            ):
                exit()
            else:
                rich.print(
                    "[bold green]Confrimed[/bold green] to continue, force re-evaluation"
                )
                args.force = True
        elif code_diff != os.popen("git diff").read().strip():
            if not Confirm.ask(
                    "[bold orange_red1]Warning[/bold orange_red1]: Code diff detected, please confirm if you want to continue"
            ):
                exit()
            else:
                rich.print(
                    "[bold green]Confrimed[/bold green] to continue, force re-evaluation"
                )
                args.force = True
    os.system(f"git rev-parse HEAD > {exp_path}/code_version.txt")
    os.system(f"git status > {exp_path}/code_status.txt")
    os.system(f"git diff > {exp_path}/code_diff.txt")
    # dump timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(f"{exp_path}/timestamp.txt", "w") as f:
        f.write(timestamp)

    # backup exemplars and test_set
    json.dump(exemplars, open(f"{exp_path}/exemplars.json", "w"), indent=4)
    json.dump(test_set, open(f"{exp_path}/test_set.json", "w"), indent=4)
    if explain_test_set is not None:
        json.dump(explain_test_set,
                  open(f"{exp_path}/explain_test_set.json", "w"),
                  indent=4)
    # dump args
    json.dump(vars(args), open(f"{exp_path}/args.json", "w"), indent=4)

    model_config = {
        "timeout": 1,
        "bit_8": args.bit_8,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "lora_weights": args.lora_weights,
        "revision": args.model_revision,
    }
    model_class = models.get(model_name, models["default"])
    rich.print(f"Using model: [bold green]{model_name}[/bold green]")
    if args.lora_weights is not None:
        lora_weights = args.lora_weights.replace("[", "\[")
        rich.print(
            f"Using lora weights: [bold green]{lora_weights}[/bold green]")
    rich.print(f"Model class: ", end="")
    rich.print(model_class)
    model = model_class(model_name, **model_config)
    rich.print(f"Model parameters: {model.get_n_parameters()}")
    rich.print(f"Model config: ")
    rich.print(model_config)
    if args.n_exemplar > 0:
        rich.print(f"Using {args.n_exemplar} exemplars ({args.exemplar_type})")
    else:
        rich.print(f"Not using exemplars")

    # explain syntactic knowledge
    if explain_test_set is not None:
        explain_iter = map(
            RequestWarpper(model, None, args.seed, None, 0, args.dry_run),
            explain_test_set)
        for instance, answers in tqdm(explain_iter,
                                      total=len(explain_test_set)):
            instance["prediction"] = {
                "explanation": answers[0],
            }

        json.dump(explain_test_set,
                  open(f"{exp_path}/prediction.explain.json", "w"),
                  indent=4)
        exit()

    # find last prediction
    if os.path.exists(f"{exp_path}/prediction.json") and not args.force:
        predictions = json.load(open(f"{exp_path}/prediction.json", "r"))
        evaluated_ids = {prediction['id'] for prediction in predictions}
        filtered_test_set = [
            question for question in test_set
            if question['id'] not in evaluated_ids
        ]
    else:
        predictions = []
        filtered_test_set = test_set

    n_workers = args.n_workers
    if args.max_eval_instances is not None:
        chunksize = min(10, args.max_eval_instances // n_workers)
    else:
        chunksize = min(10, len(filtered_test_set) // n_workers)

    exact_match = {"yes_no": 0, "multiple_choice": 0, "fill_in_the_blank": 0}
    macro_f1_score = {"f1_score": 0, "p_score": 0, "r_score": 0}
    true_positive = 0
    predict_word_count = 0
    correct_word_count = 0
    n_questions = {"yes_no": 0, "multiple_choice": 0, "fill_in_the_blank": 0}
    n_incorrect_format = {
        "yes_no": 0,
        "multiple_choice": 0,
        "fill_in_the_blank": 0
    }
    n_evaluated = 0
    # restore state
    for instance in predictions:
        question_type = instance["type"]
        n_questions[question_type] += 1
        if question_type == "yes_no":
            if instance["prediction"]["correctness"]:
                exact_match["yes_no"] += 1
        elif question_type == "multiple_choice":
            if instance["prediction"]["correctness"]:
                exact_match["multiple_choice"] += 1
        elif question_type == "fill_in_the_blank":
            macro_f1_score["f1_score"] += instance["prediction"]["f1_score"]
            macro_f1_score["p_score"] += instance["prediction"]["p_score"]
            macro_f1_score["r_score"] += instance["prediction"]["r_score"]
            true_positive += instance["prediction"]["tp"]
            predict_word_count += instance["prediction"]["p_count"]
            correct_word_count += instance["prediction"]["g_count"]
            if instance["prediction"]["tp"] == instance["prediction"][
                    "g_count"] == instance["prediction"]["p_count"]:
                exact_match["fill_in_the_blank"] += 1
        n_evaluated += 1

    # check
    if len(filtered_test_set) == 0:
        rich.print(
            "[bold green]All[/bold green] questions have been evaluated.")
    else:
        with Pool(n_workers) as p:
            try:
                if n_workers > 1 and not args.debug_mode and isinstance(
                        model, ChatGPTModel) or isinstance(model, RandomModel):
                    # parallel evaluation for ChatGPTModel
                    eval_iter = p.imap_unordered(RequestWarpper(
                        model, exemplars, args.seed, args.exemplar_type,
                        args.n_exemplar, args.dry_run),
                                                 filtered_test_set,
                                                 chunksize=chunksize)
                else:
                    eval_iter = map(
                        RequestWarpper(model, exemplars, args.seed,
                                       args.exemplar_type, args.n_exemplar,
                                       args.dry_run), filtered_test_set)
                for instance, answers in tqdm(eval_iter,
                                              total=len(test_set),
                                              initial=len(predictions)):
                    if args.max_eval_instances is not None and n_evaluated >= args.max_eval_instances:
                        break

                    answer = answers[0]
                    question_type = instance["type"]
                    n_questions[instance["type"]] += 1
                    original_answer = copy.deepcopy(answer)
                    if question_type == "yes_no":
                        answer = get_possible_answers(answer, question_type)
                        if answer is None:
                            instance["prediction"] = {
                                "answer": original_answer,
                                "eval_answer": "<Wrong format>",
                                "correctness": False
                            }
                            n_incorrect_format[question_type] += 1
                        else:
                            answer = answer[0]
                            if answer.lower().startswith(instance["answer"]):
                                exact_match["yes_no"] += 1
                                is_correct = True
                            else:
                                is_correct = False
                            # record pediction
                            instance["prediction"] = {
                                "answer": original_answer,
                                "eval_answer": answer,
                                "correctness": is_correct
                            }
                    elif question_type == "multiple_choice":
                        answer = get_possible_answers(answer, question_type)
                        if answer is None:
                            instance["prediction"] = {
                                "answer": original_answer,
                                "eval_answer": "<Wrong format>",
                                "correctness": False
                            }
                            n_incorrect_format[question_type] += 1
                        else:
                            answer = answer[0]
                            correct_answer = option_label[int(
                                instance["correct_choice"])]
                            if answer.startswith(correct_answer):
                                exact_match["multiple_choice"] += 1
                                is_correct = True
                            else:
                                is_correct = False
                                if answer[0].lower() not in "abcd":
                                    n_incorrect_format["multiple_choice"] += 1

                            # record pediction
                            instance["prediction"] = {
                                "answer": original_answer,
                                "eval_answer": answer,
                                "correctness": is_correct
                            }
                    elif question_type == "fill_in_the_blank":
                        # calculate F1 score
                        possible_answers = get_possible_answers(
                            answer,
                            question_type,
                            knowledge_point=instance["knowledge_point"])
                        answers = [tokenize(a) for a in possible_answers]
                        correct_answer_words = tokenize(instance["answer"])
                        g_count = len(correct_answer_words)
                        g_count = g_count if g_count > 0 else 1e-8

                        max_f1_score = 0
                        min_p_count = 100000000
                        max_tp = 0

                        for answer_words in answers:
                            p_count = len(answer_words)
                            tp = len(
                                lcs(answer_words, correct_answer_words)[0])

                            p_count = p_count if p_count > 0 else 1e-8
                            f1_score = (2 * tp / (p_count + g_count))
                            if f1_score > max_f1_score:
                                max_f1_score = f1_score
                                min_p_count = p_count
                                max_tp = tp
                            elif f1_score == max_f1_score and p_count < min_p_count:
                                min_p_count = p_count

                        f1_score = max_f1_score
                        p_count = min_p_count
                        tp = max_tp

                        p_score = (tp / p_count)
                        r_score = (tp / g_count)
                        macro_f1_score["f1_score"] += f1_score
                        macro_f1_score["p_score"] += p_score
                        macro_f1_score["r_score"] += r_score
                        true_positive += tp
                        predict_word_count += p_count
                        correct_word_count += g_count
                        if tp == g_count == p_count:
                            exact_match["fill_in_the_blank"] += 1
                        instance["prediction"] = {
                            "answer": original_answer,
                            "eval_answer": possible_answers,
                            "f1_score": f1_score,
                            "p_score": p_score,
                            "r_score": r_score,
                            "tp": tp,
                            "p_count": p_count,
                            "g_count": g_count
                        }

                    predictions.append(instance)
                    n_evaluated += 1
                    if n_evaluated % args.save_results_per == 0:
                        json.dump(predictions,
                                  open(f"{exp_path}/prediction.json", "w"),
                                  indent=4)

            except KeyboardInterrupt:
                rich.print(
                    f"[bold red]KeyboardInterrupt[/bold red] detected, stopping evaluation, and saving results ..."
                )

    # re-order predictions
    prediction_dict = {
        prediction['id']: prediction
        for prediction in predictions
    }
    sorted_predictions = [
        prediction_dict[question['id']] for question in test_set
        if question['id'] in prediction_dict
    ]

    json.dump(sorted_predictions,
              open(f"{exp_path}/prediction.json", "w"),
              indent=4)
    logger(f"Model:                           {model_name}")
    if args.model_revision != "main":
        logger(f"Revision:                        {args.model_revision}")
    logger(
        f"Exact match (     Yes / No      ): ({exact_match['yes_no']:0>6d} / {n_questions['yes_no']:0>6d}) = {exact_match['yes_no'] / (n_questions['yes_no'] + 1e-8):7.2%}"
    )
    logger(
        f"Exact match (  Multiple Choice  ): ({exact_match['multiple_choice']:0>6d} / {n_questions['multiple_choice']:0>6d}) = {exact_match['multiple_choice'] / (n_questions['multiple_choice'] + 1e-8):7.2%}"
    )
    logger(
        f"F1 (M-avg)  ╭ Fill-in-the-blank ╮:                     {macro_f1_score['f1_score'] / (n_questions['fill_in_the_blank']  + 1e-8):7.2%}"
    )
    logger(
        f"P  (M-avg)  │ ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ │:                     {macro_f1_score['p_score'] / (n_questions['fill_in_the_blank']  + 1e-8):7.2%}"
    )
    logger(
        f"R  (M-avg)  │ ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ │:                     {macro_f1_score['r_score'] / (n_questions['fill_in_the_blank']  + 1e-8):7.2%}"
    )
    logger(
        f"F1 (m-avg)  │ ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ │:                     {2 * true_positive / (predict_word_count + correct_word_count + 1e-8):7.2%}"
    )
    logger(
        f"P  (m-avg)  │ ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ │:                     {true_positive / (predict_word_count + 1e-8):7.2%}"
    )
    logger(
        f"R  (m-avg)  │ ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ │:                     {true_positive / (correct_word_count + 1e-8):7.2%}"
    )
    logger(
        f"Exact match ╰───────────────────╯: ({exact_match['fill_in_the_blank']:0>6d} / {n_questions['fill_in_the_blank']:0>6d}) = {exact_match['fill_in_the_blank'] / (n_questions['fill_in_the_blank'] + 1e-8):7.2%}"
    )
    logger("------------------------------------------------------------")
    logger(
        f"Incorrect format (     Yes / No      ): ({n_incorrect_format['yes_no']:0>6d} / {n_questions['yes_no']:0>6d}) = {n_incorrect_format['yes_no'] / (n_questions['yes_no'] + 1e-8):7.2%}"
    )
    logger(
        f"Incorrect format (  Multiple Choice  ): ({n_incorrect_format['multiple_choice']:0>6d} / {n_questions['multiple_choice']:0>6d}) = {n_incorrect_format['multiple_choice'] / (n_questions['multiple_choice'] + 1e-8):7.2%}"
    )
    logger(
        f"Incorrect format ( Fill-in-the-blank ): ({n_incorrect_format['fill_in_the_blank']:0>6d} / {n_questions['fill_in_the_blank']:0>6d}) = {n_incorrect_format['fill_in_the_blank'] / (n_questions['fill_in_the_blank'] + 1e-8):7.2%}"
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--task-name",
                      type=str,
                      default="syneval",
                      choices=["syneval", "mmlu"],
                      help="The evaluation task.")
    args.add_argument("--model-name",
                      type=str,
                      required=True,
                      help="The name of the model to use.")
    args.add_argument("--model-revision",
                      type=str,
                      default='main',
                      help="The revision of the model to use.")
    args.add_argument("--lora-weights",
                      type=str,
                      default=None,
                      help="The path to the lora weights.")
    args.add_argument("--bit-8",
                      action="store_true",
                      help="Use 8-bit precision.")
    args.add_argument("--sample-size",
                      type=str,
                      default='5',
                      help="The sample size for test set.")
    args.add_argument("--n-exemplar",
                      type=int,
                      default=5,
                      help="The number of exemplars to use.")
    args.add_argument("--seed", type=int, default=42, help="Random seed.")
    args.add_argument("--suite",
                      type=str,
                      required=True,
                      help="Name of the suite this run belongs to.")
    args.add_argument("--n-workers",
                      type=int,
                      default=None,
                      help="Number of workers.")
    args.add_argument("--force",
                      action="store_true",
                      help="Force re-evaluation.")
    args.add_argument("--ignore-code-revision",
                      action="store_true",
                      help="Ignore code revision.")
    args.add_argument("--save-results-per",
                      type=int,
                      default=25,
                      help="Save results per x evalation.")
    args.add_argument("--exemplar-type",
                      type=str,
                      default="all",
                      choices=[
                          "all", "generation-method", "syntactic-knowledge",
                          "exclude-self-syntactic-knowledge", "subject"
                      ],
                      help="The type of exemplars to use.")
    args.add_argument("--fine-grained-exemplars",
                      action="store_true",
                      help="Use fine-grained exemplars.")
    args.add_argument("--explain-syntactic-knowledge",
                      action="store_true",
                      help="Explain syntactic knowledge.")
    args.add_argument("--debug-mode",
                      action="store_true",
                      help="Run in debug mode.")
    args.add_argument(
        "-m",
        "--max-eval-instances",
        type=int,
        default=None,
        help="Maximum number of instances to evaluate on",
    )
    args.add_argument("--dry-run",
                      action="store_true",
                      help="Dry run, do not actually request model.")
    args = args.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
