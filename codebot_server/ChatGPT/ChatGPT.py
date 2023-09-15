"""
A simple wrapper for the official ChatGPT API
"""
import argparse
import ast
import asyncio
import json
import os
import sys
from typing import NoReturn
from litellm import acompletion

import requests
import tiktoken

from .utils import create_completer
from .utils import create_session
from .utils import get_input
import openai

ENGINE = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo"


class Chatbot:
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        engine: str = None,
        proxy: str = None,
        max_tokens: int = 4095,
        temperature: float = 0.5,
        top_p: float = 1.0,
        reply_count: int = 1,
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        function_manager=None,
        max_tier=10,
        session_id=None,
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        self.session_id = session_id
        self.max_tier = max_tier
        self.tier = 0
        self.engine = engine or ENGINE
        self.session = requests.Session()
        self.api_key = api_key
        self.function_manager = function_manager
        self.system_prompt = system_prompt
        openai.api_key = self.api_key
        if base_url is not None:
            if base_url != '':
                #  如果最后是/去掉
                if base_url.endswith('/'):
                    base_url = base_url[:-1]
                if base_url.endswith('/v1'):
                    openai.api_base = base_url
                else:
                    openai.api_base = base_url + '/v1'
        self.base_url = openai.api_base
        self.proxy = proxy
        if self.proxy:
            proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            self.session.proxies = proxies
        self.load()
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reply_count = reply_count

        if self.get_token_count("default") > self.max_tokens:
            raise Exception("System prompt is too long")

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
    ) -> None:
        """
        Add a message to the conversation
        """
        self.conversation[convo_id].append({"role": role, "content": message})

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.get_token_count(convo_id) > self.max_tokens
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                self.conversation[convo_id].pop(1)
            else:
                break

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        # if self.engine not in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", 'gpt-4-0314', 'gpt-4', 'gpt-4-32k']:
        #     raise NotImplementedError("Unsupported engine {self.engine}")

        encoding = tiktoken.encoding_for_model(self.engine)

        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                if value is None:
                    continue
                if isinstance(value, dict):  # Convert dictionary values to strings
                    value = str(value)
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def get_max_tokens(self, convo_id: str) -> int:
        """
        Get max tokens
        """
        return 4000 - self.get_token_count(convo_id)
    
    async def recursive_part(self, convo_id, **kwargs):
        if self.tier < self.max_tier:
            async for result in self.ask_stream(prompt=None, convo_id=convo_id, add_tier=True, **kwargs):
                yield result

    async def ask_stream(
        self,
        prompt: str = None,
        role: str = "user",
        convo_id: str = "default",
        add_tier: bool = False,
        **kwargs,
    ) -> str:
        """
        Ask a question
        """
        # Make conversation if it doesn't exist
        if add_tier:
            self.tier += 1
        else:
            self.tier = 0
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        if prompt not in [None, ""]:
            self.add_to_conversation(prompt, "user", convo_id=convo_id)
            self.__truncate_conversation(convo_id=convo_id)
        if self.function_manager:
            response = await acompletion(
                api_key=self.api_key,
                api_base=self.base_url,
                model=self.engine,
                messages=self.conversation[convo_id],
                functions=self.function_manager.generate_functions_array(),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                n=kwargs.get("n", self.reply_count),
                stream=True,
                user=role,
                function_call="auto",
            )
        else:
            response = await acompletion(
                api_key=self.api_key,
                api_base=self.base_url,
                model=self.engine,
                messages=self.conversation[convo_id],
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                n=kwargs.get("n", self.reply_count),
                stream=True,
                user=role,
            )  
        response_role: str = None
        full_response: str = ""
        function_name = None
        function_args = ""
        for resp in response:
            choices = resp.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta")
            if not delta:
                yield resp
                continue
            if "role" in delta:
                response_role = delta["role"]
            else:
                response_role = "assistant"
            if "function_call" in delta:
                if "name" in delta["function_call"]:
                    function_name = delta["function_call"]["name"]
                if "arguments" in delta["function_call"]:
                    function_args += delta["function_call"]["arguments"]
            else:
                if "content" in delta:
                    content = delta["content"]
                    full_response += content
            yield resp
        if function_name and function_args:
            function_request = { "role": "assistant", "content": None, "function_call": { "name": function_name, "arguments": function_args } }
            try:
                arguments = json.loads(function_args)
            except:
                try:
                    arguments = ast.literal_eval(function_args)
                except:
                    if function_name == 'python' or function_name == 'python_exec':
                        if function_name == 'python':
                            function_name = 'python_exec'
                        arguments = {"code": function_args}
                        function_args = json.dumps(arguments)
                        function_request = { "role": "assistant", "content": None, "function_call": { "name": function_name, "arguments": function_args } }
            def call_async_function():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                args = json.loads(function_args)
                if function_name == 'python_exec':
                    args['session_id'] = self.session_id
                    print(args)
                return loop.run_until_complete(self.function_manager.call_function(function_name, args))
            try:
                function_response = call_async_function()
            except Exception as e:
                function_response = { "role": "function", "name": function_name, "content": str(e) }
            if isinstance(function_response, (list, tuple, dict)):
                function_response = json.dumps(function_response)
            yield {"choices": [{"delta": {"tool": function_name, "output": function_response}, "finish_reason": "output"}]}
            self.conversation[convo_id].append(function_request)
            self.conversation[convo_id].append({"role": "function", "name": function_name, "content": function_response})
            if self.tier < self.max_tier:
                async for result in self.recursive_part(convo_id, **kwargs):
                    yield result
        else:
            self.add_to_conversation(full_response, response_role, convo_id=convo_id)
            self.save()

    def ask(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        response = self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            **kwargs,
        )
        full_response: str = "".join(response)
        return full_response
    
    
    def ask_bard(
        self,
        prompt: str,
        convo_id: str = "default",
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        response = self.ask_stream(
            prompt=prompt,
            convo_id=convo_id,
            **kwargs,
        )
        full_response: str = "".join(response)
        return full_response

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {"role": "system", "content": system_prompt or self.system_prompt},
        ]

    def save(self, *convo_ids: str) -> bool:
        """
        Save the conversation to a JSON file
        """
        if os.path.exists("history") is False:
            os.mkdir("history")
        file = "history/" + self.session_id + ".json"
        try:
            with open(file, "w", encoding="utf-8") as f:
                if convo_ids:
                    json.dump({k: self.conversation[k] for k in convo_ids}, f, indent=2)
                else:
                    for convo_id in self.conversation:
                        system_message = self.conversation[convo_id][0]
                        # 只保留50条对话，但是必须要保留系统的第一条消息
                        self.conversation[convo_id] = self.conversation[convo_id][-50:]
                        self.conversation[convo_id][0] = system_message
                    json.dump(self.conversation, f, indent=2)
        except (FileNotFoundError, KeyError):
            return False
        return True
        # print(f"Error: {file} could not be created")

    def load(self, *convo_ids: str) -> bool:
        """
        Load the conversation from a JSON  file
        """
        if os.path.exists("history") is False:
            os.mkdir("history")
        file = "history/" + self.session_id + ".json"
        try:
            with open(file, encoding="utf-8") as f:
                if convo_ids:
                    convos = json.load(f)
                    self.conversation.update({k: convos[k] for k in convo_ids})
                else:
                    self.conversation = json.load(f)
        except (FileNotFoundError, KeyError, json.decoder.JSONDecodeError):
            self.conversation: dict = {
                "default": [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                ],
            }
            self.save()
            return False
        return True
    
    def voice_2_text(self, file):
        # 判断file是文件还是文件的字符串，如果是文件的字符串，就转换成文件
        if isinstance(file, str):
            file = open(file, 'rb')
        response = openai.Audio.transcribe('whisper-1', file)
        return response['text']
