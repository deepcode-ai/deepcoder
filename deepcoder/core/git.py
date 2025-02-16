import shutil
import subprocess

from pathlib import Path
from typing import List

from deepcoder.core.files_dict import FilesDict


def is_git_installed() -> bool:
    return shutil.which("git") is not None


def is_git_repo(path: Path) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).returncode
        == 0
    )


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path)


def has_uncommitted_changes(path: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "diff", "--exit-code"],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).returncode
    )


def filter_files_with_uncommitted_changes(
    basepath: Path, files_dict: FilesDict
) -> List[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only"], cwd=basepath, stdout=subprocess.PIPE, text=True
    )
    files_with_diff = result.stdout.splitlines()
    return [f for f in files_dict.keys() if f in files_with_diff]


def stage_files(path: Path, files: List[str]) -> None:
    subprocess.run(["git", "add", *files], cwd=path, check=True)


def filter_by_gitignore(path: Path, file_list: List[str]) -> List[str]:
    if not file_list:
        return []

    result = subprocess.run(
        ["git", "-C", ".", "check-ignore", "--no-index", "--stdin"],
        cwd=path,
        input="\n".join(file_list).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ignored_paths = result.stdout.splitlines()
    return [f for f in file_list if f not in ignored_paths]


def stage_uncommitted_to_git(
    path: Path, files_dict: FilesDict, improve_mode: bool
) -> None:
    if is_git_installed() and not improve_mode:
        if not is_git_repo(path):
            print("\nInitializing an empty git repository")
            init_git_repo(path)

    if is_git_repo(path):
        modified_files = filter_files_with_uncommitted_changes(path, files_dict)
        if modified_files:
            print(
                "Staging the following uncommitted files before overwriting: ",
                ", ".join(str(f) for f in modified_files),
            )
            stage_files(path, [str(f) for f in modified_files])
