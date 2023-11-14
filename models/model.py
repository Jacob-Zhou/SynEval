import copy
import math
import os
import random
import rich
import torch

from utils.fn import upper_first_letter
from utils.common import option_label

opened_questions = {"fill_in_the_blank", "short_answer"}


class Model:

    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name

    def get_answer(self, *args, **kwargs):
        raise NotImplementedError

    def qusetion_to_prompt(self, instance, with_answer=False, *args, **kwargs):
        raise NotImplementedError

    def build_request(self, exemplars, instance, seed, *args, **kwargs):
        raise NotImplementedError

    def get_request_prefix(self,
                           instance,
                           is_zero_shot=False,
                           *args,
                           **kwargs):
        question_type = instance['type']
        prefix = ""
        # prefix = "You are an English as a second language (ESL) student. You are unfamiliar with the English grammar. You are trying to answer the following questions. "
        if question_type == "yes_no":
            prefix += f"The following are true or false questions"
        elif question_type == "multiple_choice":
            prefix += f"The following are multiple choice questions"
        elif question_type == "fill_in_the_blank":
            prefix += f"The following are fill-in-the-blank questions"
        elif question_type == "short_answer":
            prefix += f"The following are short answer questions"
        else:
            raise NotImplementedError
        if is_zero_shot:
            if question_type == "yes_no":
                prefix += f", please answer them with “True” or “False”.\n"
            elif question_type == "multiple_choice":
                prefix += f", please answer them with “A”, “B”, “C”, or “D”.\n"
            elif question_type == "fill_in_the_blank":
                prefix += f".\n"
            elif question_type == "short_answer":
                prefix += f", please answer them in a sentence.\n"
            else:
                raise NotImplementedError
        else:
            prefix += " (with answers):\n"
        return prefix

    def get_output_prefix(self, *args, **kwargs):
        return "Answer: The answer is “"

    def get_n_parameters(self):
        return "N/A"


class ParameterizedModel(object):

    def get_n_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        all_param = 0
        if self.model is None:
            return "N/A"
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params

        # convert to human readable format (i.e. 178B instead of 178000000000)
        def human_format(num):
            num = int(num)
            if num == 0:
                return "0"
            units = [
                "", "K", "M", "B", "T", "P", "E", "Z", "Y", "B", "C", "D", "N",
                "U"
            ]
            p = int(math.floor(math.log(num) / math.log(1000)))
            s = round(num / math.pow(1000, p), 2)
            return "%s%s" % (s, units[p])

        return human_format(all_param)


class ChatModel(object):
    pass


class ChatGPTModel(Model, ChatModel):

    def __init__(self, model_name, timeout=1, *args, **kwargs) -> None:
        super().__init__(model_name, *args, **kwargs)
        # Load your API key from an environment variable or secret management service
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self.init_prompt = [
            {
                "role": "system",
                "content": "You are an English master."
            },
        ]
        self.create = openai.ChatCompletion.create

    def get_answer(self, prompts, question_type):
        # gpt-4, gpt-4-0314, gpt-3.5-turbo
        response = self.create(
            model=self.model_name,
            messages=self.init_prompt + prompts,
            temperature=0.0,
            n=1,
            timeout=self.timeout,
            max_tokens=256 if question_type in opened_questions else 10)
        answers = [
            choice.message['content'].strip() for choice in response.choices
        ]
        return answers

    def qusetion_to_prompt(self, instance, with_answer=False):
        chat_list = []
        question_type = instance["type"]
        question_text = ""
        if 'sentence' in instance:
            question_text += f"Sentence: {instance['sentence']}\n"
        question_text += f"Question: {instance['question']}"
        if question_type in {"yes_no", "fill_in_the_blank", "short_answer"}:
            chat_list = [{"role": "user", "content": question_text}]
            if with_answer:
                chat_list.append({
                    "role":
                    "assistant",
                    "content":
                    f"Answer: The answer is “{upper_first_letter(instance['answer'])}”"
                })
        elif question_type == "multiple_choice":
            question_text += "\n"
            choices = instance["options"]
            for i, choice in enumerate(choices):
                try:
                    question_text += f"{option_label[i]}. {upper_first_letter(choice)}\n"
                except:
                    rich.print(instance)
                    raise Exception
            chat_list = [{"role": "user", "content": question_text}]
            if with_answer:
                answer = int(instance["correct_choice"])
                chat_list.append({
                    "role":
                    "assistant",
                    "content":
                    f"Answer: The answer is “{option_label[answer]}.”"
                })
        return chat_list

    def build_request(self, exemplars, instance, *args, **kwargs):
        chat_list = [{
            "role":
            "user",
            "content":
            self.get_request_prefix(instance, is_zero_shot=len(exemplars) == 0)
        }]
        for examplar in exemplars:
            examplar_text = self.qusetion_to_prompt(examplar, with_answer=True)
            chat_list += examplar_text
        prompt = self.qusetion_to_prompt(instance, with_answer=False)
        chat_list += prompt
        return chat_list


class HuggingFaceModel(ParameterizedModel, Model):

    def __init__(self,
                 model_name,
                 lora_weights=None,
                 bit_8=False,
                 timeout=1,
                 stop_sequences="\n",
                 dry_run=False,
                 revision='main',
                 *args,
                 **kwargs) -> None:
        super().__init__(model_name, *args, **kwargs)
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        self.model_name = model_name
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       padding_side="left",
                                                       revision=revision,
                                                       trust_remote_code=True)
        self.max_new_tokens = 256

        if dry_run:
            self.model = None
        else:
            if bit_8:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=True,
                    revision=revision,
                    trust_remote_code=True)
            else:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        revision=revision,
                        trust_remote_code=True)
                except:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        revision=revision,
                    )

            if lora_weights is not None:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_weights,
                    torch_dtype=torch.float16,
                    load_in_8bit=bit_8)

            self.model.eval()

        # force end with stop_sequences
        stop_sequence_ids = self.tokenizer(stop_sequences,
                                           return_token_type_ids=False,
                                           add_special_tokens=False)
        self.eos_token_id = stop_sequence_ids.input_ids[-1]
        self.pad_token_id = self.tokenizer.unk_token_id or self.eos_token_id

        self.generation_config = GenerationConfig(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            max_new_tokens=self.max_new_tokens)

    def get_answer(self, prompt, question_type):
        input_prompt = prompt + self.get_output_prefix()
        prompt_input_ids = self.tokenizer(prompt,
                                          return_tensors="pt",
                                          return_token_type_ids=False)
        tokenized_inputs = self.tokenizer(
            input_prompt, return_tensors="pt",
            return_token_type_ids=False).to('cuda')

        generation_config = copy.deepcopy(self.generation_config)
        generation_config.update(
            max_new_tokens=self.max_new_tokens if question_type in
            opened_questions else 1)

        with torch.no_grad():
            generate_ids = self.model.generate(
                **tokenized_inputs, generation_config=generation_config)

        prompt_len = prompt_input_ids.input_ids.shape[-1]
        outputs = self.tokenizer.batch_decode(generate_ids[:, prompt_len:],
                                              skip_special_tokens=True)
        answer = outputs[0].strip()
        return [answer]

    def qusetion_to_prompt(self, instance, with_answer=False):
        prompt = ''
        question_type = instance["type"]
        if 'sentence' in instance:
            prompt += f"Sentence: {instance['sentence']}\n"
        prompt += f"Question: {instance['question']}\n"
        # prompt += f"Sentence: {instance['sentence']}\nQuestion: {instance['question']}\n"
        if question_type in {"yes_no", "fill_in_the_blank", "short_answer"}:
            if with_answer:
                prompt += f"Answer: The answer is “{upper_first_letter(instance['answer'])}”\n"
        elif question_type == "multiple_choice":
            choices = instance["options"]
            for i, choice in enumerate(choices):
                prompt += f"{option_label[i]}. {upper_first_letter(choice)}\n"
            if with_answer:
                answer = int(instance["correct_choice"])
                prompt += f"Answer: The answer is “{option_label[answer]}.”\n"
        return prompt

    def build_request(self, exemplars, instance, *args, **kwargs):
        prompt = self.get_request_prefix(instance,
                                         is_zero_shot=len(exemplars) == 0)
        for examplar in exemplars:
            examplar_text = self.qusetion_to_prompt(examplar, with_answer=True)
            prompt += examplar_text
        prompt += self.qusetion_to_prompt(instance, with_answer=False)
        return prompt


class FalconInstructModel(HuggingFaceModel):

    def __init__(self, model_name, *args, **kwargs) -> None:
        assert model_name in {
            "tiiuae/falcon-40b-instruct", "tiiuae/falcon-7b-instruct"
        }
        super().__init__(model_name, *args, **kwargs)
        stop_sequence = ["”", ".”", "”.", ",”", "”,", "\n"]
        self.eos_token_id = [
            self.tokenizer(stop,
                           return_token_type_ids=False,
                           add_special_tokens=False).input_ids[-1]
            for stop in stop_sequence
        ]
        from transformers import GenerationConfig
        self.generation_config = GenerationConfig(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            max_new_tokens=self.max_new_tokens)


class LlamaChatModel(HuggingFaceModel):

    def __init__(self, model_name, *args, **kwargs) -> None:
        assert model_name in {
            "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1"
        }
        self.B_INST = "[INST]"
        self.E_INST = "[/INST]"
        super().__init__(model_name, *args, **kwargs)
        self.stop_ids = [self.eos_token_id, self.tokenizer.eos_token_id]
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.unk_token_id or self.eos_token_id

        self.generation_config.update(eos_token_id=self.stop_ids,
                                      pad_token_id=self.pad_token_id)

    def get_request_prefix(self, instance, is_zero_shot, *args, **kwargs):
        # sys_prompt = "<<SYS>>\nYou are an English as a second language (ESL) student. You are unfamiliar with the English grammar.\n<</SYS>>\n\n"
        sys_prompt = "<<SYS>>\nYou are an English master.\n<</SYS>>\n\n"
        request_prefix = sys_prompt + super().get_request_prefix(
            instance, is_zero_shot, *args, **kwargs)
        return f"{self.B_INST} {request_prefix.strip()} {self.E_INST} "

    def qusetion_to_prompt(self, instance, with_answer=False):
        prompt = f"{self.B_INST} "
        question_type = instance["type"]
        if 'sentence' in instance:
            prompt += f"Sentence: {instance['sentence']}\n"
        prompt += f"Question: {instance['question']}"
        # prompt += f"Sentence: {instance['sentence']}\nQuestion: {instance['question']}"
        if question_type in {"yes_no", "fill_in_the_blank", "short_answer"}:
            prompt += f" {self.E_INST} "
            if with_answer:
                prompt += f"Answer: The answer is “{upper_first_letter(instance['answer'])}” "
        elif question_type == "multiple_choice":
            prompt += "\n"
            choices = instance["options"]
            for i, choice in enumerate(choices):
                prompt += f"{option_label[i]}. {upper_first_letter(choice)}\n"
            prompt += f" {self.E_INST} "
            if with_answer:
                answer = int(instance["correct_choice"])
                prompt += f"Answer: The answer is “{option_label[answer]}.”"
        return prompt

    def get_answer(self, prompts, question_type):
        bos_tenser = torch.tensor([[self.bos_token_id]])
        eos_tenser = torch.tensor([[self.eos_token_id]])
        prompt_input_ids = [
            torch.cat([
                bos_tenser,
                self.tokenizer(prompt,
                               return_tensors="pt",
                               return_token_type_ids=False,
                               add_special_tokens=False).input_ids, eos_tenser
            ],
                      dim=-1) for prompt in prompts[:-1]
        ]
        last_prompt_input_ids = self.tokenizer(
            prompts[-1], return_tensors="pt",
            return_token_type_ids=False).input_ids
        output_prefix_ids = self.tokenizer(self.get_output_prefix().strip(),
                                           return_tensors="pt",
                                           return_token_type_ids=False,
                                           add_special_tokens=False).input_ids
        tokenized_inputs = torch.cat(
            [*prompt_input_ids, last_prompt_input_ids, output_prefix_ids],
            dim=-1).to('cuda')

        generation_config = copy.deepcopy(self.generation_config)
        generation_config.update(
            max_new_tokens=self.max_new_tokens if question_type in
            opened_questions else 1)

        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids=tokenized_inputs,
                generation_config=generation_config)

        prompt_len = tokenized_inputs.shape[-1] - output_prefix_ids.shape[-1]
        outputs = self.tokenizer.batch_decode(generate_ids[:, prompt_len:],
                                              skip_special_tokens=True)
        answer = outputs[0].strip()
        return [answer]

    def build_request(self, exemplars, instance, *args, **kwargs):
        prompt = [
            self.get_request_prefix(instance, is_zero_shot=len(exemplars) == 0)
        ]
        for examplar in exemplars:
            examplar_text = self.qusetion_to_prompt(examplar, with_answer=True)
            prompt += [examplar_text]
        prompt += [self.qusetion_to_prompt(instance, with_answer=False)]
        return prompt


class GlmChatModel(HuggingFaceModel, ChatModel):

    def __init__(self, model_name, *args, **kwargs) -> None:
        assert model_name in {"THUDM/chatglm2-6b"}
        super().__init__(model_name, *args, **kwargs)

    def get_answer(self, prompts, question_type):
        history = prompts[:-1]
        inputs = self.model.build_inputs(self.tokenizer,
                                         prompts[-1][0],
                                         history=history)
        outputs = self.model.generate(**inputs,
                                      max_new_tokens=(
                                          self.max_new_tokens if question_type
                                          in opened_questions else 10),
                                      num_beams=1,
                                      do_sample=False,
                                      top_p=1.0,
                                      temperature=1.0)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(outputs)
        response = self.model.process_response(response)
        return [response.strip()]

    def qusetion_to_prompt(self, instance, with_answer=False):
        chat_list = []
        question_type = instance["type"]
        question_text = ''
        if 'sentence' in instance:
            question_text += f"Sentence: {instance['sentence']}\n"
        question_text += f"Question: {instance['question']}"
        # question_text = f"Sentence: {instance['sentence']}\nQuestion: {instance['question']}\n"
        if question_type in {"yes_no", "fill_in_the_blank", "short_answer"}:
            chat_list = [(
                question_text,
                f"Answer: The answer is “{upper_first_letter(instance['answer'])}”"
            )]
        elif question_type == "multiple_choice":
            question_text += "\n"
            choices = instance["options"]
            for i, choice in enumerate(choices):
                try:
                    question_text += f"{option_label[i]}. {upper_first_letter(choice)}\n"
                except:
                    rich.print(instance)
                    raise Exception
            answer = int(instance["correct_choice"])
            chat_list = [(question_text,
                          f"Answer: The answer is “{option_label[answer]}.”")]
        if with_answer:
            return chat_list
        else:
            return [(chat_list[0][0], None)]

    def build_request(self, exemplars, instance, *args, **kwargs):
        chat_list = [
            (self.get_request_prefix(instance,
                                     is_zero_shot=len(exemplars) == 0),
             "I'm happy to answer your questions.")
        ]
        for examplar in exemplars:
            examplar_text = self.qusetion_to_prompt(examplar, with_answer=True)
            chat_list += examplar_text
        prompt = self.qusetion_to_prompt(instance, with_answer=False)
        chat_list += prompt
        return chat_list


class BaichuanChatModel(ParameterizedModel, ChatGPTModel):

    def __init__(self,
                 model_name,
                 lora_weights=None,
                 bit_8=False,
                 timeout=1,
                 stop_sequences="\n",
                 dry_run=False,
                 *args,
                 **kwargs) -> None:
        super().__init__(model_name, *args, **kwargs)
        assert model_name in {
            "baichuan-inc/Baichuan2-7B-Chat", "baichuan-inc/Baichuan2-13B-Chat"
        }
        self.timeout = timeout
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_fast=False,
                                                       trust_remote_code=True)
        self.max_new_tokens = 256

        if dry_run:
            self.model = None
        else:
            if bit_8:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=True,
                    trust_remote_code=True)
            else:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True)
                except:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16)

            if lora_weights is not None:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_weights,
                    torch_dtype=torch.float16,
                    load_in_8bit=bit_8)

            self.model.eval()
            self.model.generation_config = GenerationConfig.from_pretrained(
                model_name)
            self.model.generation_config.update(
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                do_sample=False,
            )

    def get_answer(self, prompts, question_type):
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.update(
            max_new_tokens=self.max_new_tokens if question_type in
            opened_questions else 10)
        response = self.model.chat(self.tokenizer,
                                   self.init_prompt + prompts,
                                   generation_config=generation_config)
        return [response.strip()]


class RandomModel(HuggingFaceModel):

    def __init__(self, model_name, seed=0, *args, **kwargs) -> None:
        self.model_name = model_name
        self.seed = seed

    def get_answer(self, prompt, question_type):
        random.seed(f"{prompt} seed@{self.seed}")
        if prompt.startswith("The following are true or false questions"):
            return [
                f"Answer: The answer is “{random.choice(['True', 'False'])}”"
            ]
        elif prompt.startswith("The following are multiple choice questions"):
            return [
                f"Answer: The answer is “{random.choice(option_label[:4])}.”"
            ]
        elif prompt.startswith(
                "The following are fill-in-the-blank questions"):
            sentence = prompt.split("\nSentence: ")[-1].split(
                "\n")[0].strip().split()
            n_words = len(sentence)
            # random cut the sentence
            start, end = sorted(random.sample(range(n_words + 1), 2))
            random_chunk = " ".join(sentence[start:end])
            return [f"Answer: The answer is “{random_chunk}”"]
        else:
            raise NotImplementedError

    def get_n_parameters(self):
        return "N/A"


models = {
    "gpt-3.5-turbo": ChatGPTModel,
    "gpt-3.5-turbo-0314": ChatGPTModel,
    "gpt-3.5-turbo-0613": ChatGPTModel,
    "gpt-4": ChatGPTModel,
    "gpt-4-0314": ChatGPTModel,
    "gpt-4-0613": ChatGPTModel,
    "meta-llama/Llama-2-7b-chat-hf": LlamaChatModel,
    "meta-llama/Llama-2-13b-chat-hf": LlamaChatModel,
    "meta-llama/Llama-2-70b-chat-hf": LlamaChatModel,
    "mistralai/Mistral-7B-Instruct-v0.1": LlamaChatModel,
    "tiiuae/falcon-7b-instruct": FalconInstructModel,
    "tiiuae/falcon-40b-instruct": FalconInstructModel,
    "baichuan-inc/Baichuan2-7B-Chat": BaichuanChatModel,
    "baichuan-inc/Baichuan2-13B-Chat": BaichuanChatModel,
    "THUDM/chatglm2-6b": GlmChatModel,
    "default": HuggingFaceModel,
    "random": RandomModel
}