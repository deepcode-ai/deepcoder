"""
FilesDict Module

This module provides a FilesDict class which is a dictionary-based container for managing code files.
It extends the standard dictionary to enforce string keys and values, representing filenames and their
corresponding code content. It also provides methods to format its contents for chat-based interaction
with an AI agent and to enforce type checks on keys and values.

Classes:
    FilesDict: A dictionary-based container for managing code files.
"""

from collections import OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Iterator, Union


class FilesDict(MutableMapping):
    """
    A dictionary-based container for managing code files.

    This class implements the MutableMapping interface to provide a dictionary-like
    container that enforces string keys and values, representing filenames and their
    corresponding code content. It provides methods to format its contents for
    chat-based interaction with an AI agent and enforces type checks on keys and values.
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, key: Union[str, Path]) -> str:
        return self._data[str(key)]

    def __setitem__(self, key: Union[str, Path], value: str):
        """
        Set the code content for the given filename, enforcing type checks on the key and value.

        Parameters
        ----------
        key : Union[str, Path]
            The filename as a key for the code content.
        value : str
            The code content to associate with the filename.

        Raises
        ------
        TypeError
            If the key is not a string or Path, or if the value is not a string.
        """
        if not isinstance(key, (str, Path)):
            raise TypeError("Keys must be strings or Path objects")
        if not isinstance(value, str):
            raise TypeError("Values must be strings")
        self._data[str(key)] = value

    def __delitem__(self, key: Union[str, Path]):
        del self._data[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_chat(self) -> str:
        """
        Formats the items of the object into a string suitable for chat display.

        Returns
        -------
        str
            A string representation of the files with line numbers.
        """
        chat_str = ""
        for file_name, file_content in self.items():
            lines_dict = file_to_lines_dict(file_content)
            chat_str += f"File: {file_name}\n"
            for line_number, line_content in lines_dict.items():
                chat_str += f"{line_number} {line_content}\n"
            chat_str += "\n"
        return f"\n{chat_str}"

    def to_log(self) -> str:
        """
        Formats the items of the object into a string suitable for log display.

        Returns
        -------
        str
            A string representation of the files without line numbers.
        """
        log_str = ""
        for file_name, file_content in self.items():
            log_str += f"File: {file_name}\n"
            log_str += file_content
            log_str += "\n"
        return log_str


def file_to_lines_dict(file_content: str) -> OrderedDict:
    """
    Converts file content into an ordered dictionary mapping line numbers to line content.

    Parameters
    ----------
    file_content : str
        The content of the file.

    Returns
    -------
    OrderedDict
        An ordered dictionary with line numbers as keys and line contents as values.
    """
    return OrderedDict(
        {
            line_number: line_content
            for line_number, line_content in enumerate(file_content.split("\n"), 1)
        }
    )
