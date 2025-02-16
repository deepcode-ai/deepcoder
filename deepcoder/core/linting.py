from typing import Dict, Optional

import black

from deepcoder.core.files_dict import FilesDict


class Linting:
    def __init__(self):
        # Dictionary to hold linting methods for different file types
        self.linters: Dict[str, callable] = {".py": self.lint_python}

    def lint_python(self, content: str, config: Optional[Dict] = None) -> str:
        """Lint Python files using the `black` library, handling all exceptions silently and logging them.
        This function attempts to format the code and returns the formatted code if successful.
        If any error occurs during formatting, it logs the error and returns the original content.

        Parameters
        ----------
        content : str
            The Python source code content to be formatted.
        config : Optional[Dict]
            Configuration options for black formatter.

        Returns
        -------
        str
            The formatted code if successful, otherwise the original content.
        """
        if config is None:
            config = {}

        try:
            # Try to format the content using black
            linted_content = black.format_str(content, mode=black.FileMode(**config))
            return linted_content
        except black.NothingChanged:
            # If nothing changed, log the info and return the original content
            print("\nInfo: No changes were made during formatting.\n")
            return content
        except Exception as error:
            # If any other exception occurs, log the error and return the original content
            print(f"\nError: Could not format due to {error}\n")
            return content

    def lint_files(
        self, files_dict: FilesDict, config: Optional[Dict] = None
    ) -> FilesDict:
        """
        Lints files based on their extension using registered linting functions.

        Parameters
        ----------
        files_dict : FilesDict
            The dictionary of file names to their respective source code content.
        config : Optional[Dict]
            A dictionary of configuration options for the linting tools.

        Returns
        -------
        FilesDict
            The dictionary of file names to their respective source code content after linting.
        """
        if config is None:
            config = {}

        for filename, content in files_dict.items():
            extension = filename[
                filename.rfind(".") :
            ].lower()  # Ensure case insensitivity
            if extension in self.linters:
                original_content = content
                linted_content = self.linters[extension](content, config)
                if linted_content != original_content:
                    print(f"Linted {filename}")
                else:
                    print(f"No changes made for {filename}")
                files_dict[filename] = linted_content
            else:
                print(f"No linter registered for {filename}")
        return files_dict
