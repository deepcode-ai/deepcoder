"""
AI Module

This module provides an AI class that interfaces with language models to perform various tasks such as
starting a conversation, advancing the conversation, and handling message serialization. It also includes
backoff strategies for handling rate limit errors from the OpenAI API.

Classes:
    AI: A class that interfaces with language models for conversation management and message serialization.
    ClipboardAI: A class that extends AI to support clipboard-based interactions.

Functions:
    serialize_messages(messages: List[Message]) -> str
        Serialize a list of messages to a JSON string.
"""

from __future__ import annotations

import json
import logging
import os

from pathlib import Path
from typing import Any, List, Optional, Union

import backoff
import openai
import pyperclip

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from deepcoder.core.token_usage import TokenUsageLog

# Type hint for a chat message
Message = Union[AIMessage, HumanMessage, SystemMessage]

# Set up logging
logger = logging.getLogger(__name__)


class AI:
    """
    A class that interfaces with language models for conversation management and message serialization.

    This class provides methods to start and advance conversations, handle message serialization,
    and implement backoff strategies for rate limit errors when interacting with the OpenAI API.

    Attributes
    ----------
    temperature : float
        The temperature setting for the language model.
    azure_endpoint : str | None
        The endpoint URL for the Azure-hosted language model.
    model_name : str
        The name of the language model to use.
    streaming : bool
        A flag indicating whether to use streaming for the language model.
    vision : bool
        A flag indicating whether the model supports vision capabilities.
    llm : BaseChatModel
        The language model instance for conversation management.
    token_usage_log : TokenUsageLog
        A log for tracking token usage during conversations.
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.1,
        azure_endpoint: Optional[str] = None,
        streaming: bool = True,
        vision: bool = False,
    ):
        """
        Initialize the AI class.

        Parameters
        ----------
        model_name : str
            The name of the model to use, defaults to "gpt-4-turbo".
        temperature : float
            The temperature to use for the model, defaults to 0.1.
        azure_endpoint : str | None
            The Azure endpoint URL if using Azure OpenAI, defaults to None.
        streaming : bool
            Whether to enable streaming responses, defaults to True.
        vision : bool
            Whether to enable vision capabilities, defaults to False.
        """
        self.temperature = temperature
        self.azure_endpoint = azure_endpoint
        self.model_name = model_name
        self.streaming = streaming
        self.vision = (
            vision
            or ("vision-preview" in model_name)
            or ("gpt-4-turbo" in model_name and "preview" not in model_name)
            or ("claude" in model_name)
        )
        self.llm = self._create_chat_model()
        self.token_usage_log = TokenUsageLog(model_name)

        logger.debug(f"Using model {self.model_name}")

    def start(self, system: str, user: Any, *, step_name: str) -> List[Message]:
        """
        Start the conversation with a system message and a user message.

        Parameters
        ----------
        system : str
            The content of the system message.
        user : Any
            The content of the user message.
        step_name : str
            The name of the step for logging purposes.

        Returns
        -------
        List[Message]
            The list of messages in the conversation.
        """
        messages: List[Message] = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        return self.next(messages, step_name=step_name)

    def _extract_content(self, content: Union[str, List[dict]]) -> str:
        """
        Extracts text content from a message, supporting both string and list types.

        Parameters
        ----------
        content : Union[str, List[dict]]
            The content of a message, which could be a string or a list.

        Returns
        -------
        str
            The extracted text content.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and content and isinstance(content[0], dict):
            return content[0].get("text", "")
        return ""

    def _collapse_text_messages(self, messages: List[Message]) -> List[Message]:
        """
        Combine consecutive messages of the same type into a single message.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to collapse.

        Returns
        -------
        List[Message]
            The list of messages after collapsing consecutive messages of the same type.
        """
        if not messages:
            return []

        collapsed_messages = []
        previous_message = messages[0]
        combined_content = self._extract_content(previous_message.content)

        for current_message in messages[1:]:
            if current_message.type == previous_message.type:
                combined_content += "\n\n" + self._extract_content(
                    current_message.content
                )
            else:
                collapsed_messages.append(
                    previous_message.__class__(content=combined_content)
                )
                previous_message = current_message
                combined_content = self._extract_content(current_message.content)

        collapsed_messages.append(previous_message.__class__(content=combined_content))
        return collapsed_messages

    def next(
        self,
        messages: List[Message],
        prompt: Optional[str] = None,
        *,
        step_name: str,
    ) -> List[Message]:
        """
        Advances the conversation by sending message history to LLM and updating with the response.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        prompt : Optional[str]
            The optional prompt to append to the conversation.
        step_name : str
            The name of the step for logging purposes.

        Returns
        -------
        List[Message]
            The updated list of messages in the conversation.
        """
        if prompt:
            messages.append(HumanMessage(content=prompt))

        logger.debug(
            "Creating a new chat completion:\n%s",
            "\n".join(m.pretty_repr() for m in messages),
        )

        if not self.vision:
            messages = self._collapse_text_messages(messages)

        response = self.backoff_inference(messages)

        self.token_usage_log.update_log(
            messages=messages,
            answer=response.content,
            step_name=step_name,
        )
        messages.append(response)
        logger.debug("Chat completion finished: %s", messages)

        return messages

    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_tries=7,
        max_time=45,
        logger=logger,
    )
    def backoff_inference(self, messages: List[Message]) -> Message:
        """
        Perform inference using the language model with exponential backoff for rate limits.

        Parameters
        ----------
        messages : List[Message]
            The messages to process.

        Returns
        -------
        Message
            The model's response message.

        Raises
        ------
        openai.RateLimitError
            If rate limit persists after max retries.
        """
        return self.llm.invoke(messages)

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        """
        Serialize a list of messages to a JSON string.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to serialize.

        Returns
        -------
        str
            The serialized messages as a JSON string.
        """
        return json.dumps(messages_to_dict(messages))

    @staticmethod
    def deserialize_messages(jsondictstr: str) -> List[Message]:
        """
        Deserialize a JSON string to a list of messages.

        Parameters
        ----------
        jsondictstr : str
            The JSON string to deserialize.

        Returns
        -------
        List[Message]
            The deserialized list of messages.
        """
        data = json.loads(jsondictstr)
        prevalidated_data = [
            {**item, "tools": {**item.get("tools", {}), "is_chunk": False}}
            for item in data
        ]
        return list(messages_from_dict(prevalidated_data))

    def _create_chat_model(self) -> BaseChatModel:
        """
        Create a chat model instance based on configuration.

        Returns
        -------
        BaseChatModel
            The configured chat model instance.
        """
        callbacks = [StreamingStdOutCallbackHandler()]

        if self.azure_endpoint:
            return AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                openai_api_version=os.getenv(
                    "OPENAI_API_VERSION", "2024-05-01-preview"
                ),
                deployment_name=self.model_name,
                openai_api_type="azure",
                streaming=self.streaming,
                callbacks=callbacks,
            )

        if "claude" in self.model_name:
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                callbacks=callbacks,
                streaming=self.streaming,
                max_tokens_to_sample=4096,
            )

        common_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "streaming": self.streaming,
            "callbacks": callbacks,
        }

        if self.vision:
            return ChatOpenAI(**common_params, max_tokens=4096)

        return ChatOpenAI(**common_params)


def serialize_messages(messages: List[Message]) -> str:
    """
    Convenience function to serialize messages using AI class method.

    Parameters
    ----------
    messages : List[Message]
        The messages to serialize.

    Returns
    -------
    str
        The serialized messages.
    """
    return AI.serialize_messages(messages)


class ClipboardAI(AI):
    """
    A specialized AI class that interfaces with the system clipboard for message handling.
    """

    def __init__(self, **_):
        """Initialize ClipboardAI instance."""
        self.vision = False
        self.token_usage_log = TokenUsageLog("clipboard_llm")

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        """
        Serialize messages in a human-readable format.

        Parameters
        ----------
        messages : List[Message]
            The messages to serialize.

        Returns
        -------
        str
            The formatted message string.
        """
        return "\n\n".join(f"{m.type}:\n{m.content}" for m in messages)

    @staticmethod
    def multiline_input() -> str:
        """
        Capture multiline input from the user.

        Returns
        -------
        str
            The captured input text.
        """
        print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
        content = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            content.append(line)
        return "\n".join(content)

    def next(
        self,
        messages: List[Message],
        prompt: Optional[str] = None,
        *,
        step_name: str,
    ) -> List[Message]:
        """
        Handle the next interaction using the clipboard.

        Parameters
        ----------
        messages : List[Message]
            The current message history.
        prompt : Optional[str]
            An optional prompt to append.
        step_name : str
            The name of the current step.

        Returns
        -------
        List[Message]
            The updated message history.
        """
        if prompt:
            messages.append(HumanMessage(content=prompt))

        logger.debug("Creating a new chat completion: %s", messages)

        msgs = self.serialize_messages(messages)
        pyperclip.copy(msgs)
        Path("clipboard.txt").write_text(msgs)
        print(
            f"Messages copied to clipboard and written to clipboard.txt, {len(msgs)} characters in total"
        )

        response = self.multiline_input()
        messages.append(AIMessage(content=response))
        logger.debug("Chat completion finished: %s", messages)

        return messages
