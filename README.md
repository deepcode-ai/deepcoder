# deepcoder

[![GitHub Repo stars](https://img.shields.io/github/stars/deepcoder-org/deepcoder?style=social)](https://github.com/deepcode-ai/deepcoder)
[![Discord Follow](https://dcbadge.vercel.app/api/server/8tcDQ89Ej2?style=flat)](https://discord.gg/8tcDQ89Ej2)
[![License](https://img.shields.io/github/license/deepcoder-org/deepcoder)](https://github.com/deepcode-ai/deepcoder/blob/main/LICENSE)
[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/deepcoder-org/deepcoder)](https://github.com/deepcode-ai/deepcoder/issues)
![GitHub Release](https://img.shields.io/github/v/release/deepcoder-org/deepcoder)
[![Twitter Follow](https://img.shields.io/twitter/follow/khulnasoft?style=social)](https://twitter.com/khulnasoft)

The OG code genereation experimentation platform!

If you are looking for the evolution that is an opinionated, managed service – check out deepcoder.app.

If you are looking for a well maintained hackable CLI for – check out aider.


deepcoder lets you:
- Specify software in natural language
- Sit back and watch as an AI writes and executes the code
- Ask the AI to implement improvements

## Getting Started

### Install deepcoder

For **stable** release:

- `python -m pip install deepcoder`

For **development**:
- `git clone https://github.com/deepcode-ai/deepcoder.git`
- `cd deepcoder`
- `poetry install`
- `poetry shell` to activate the virtual environment

We actively support Python 3.10 - 3.12. The last version to support Python 3.8 - 3.9 was [0.2.6](https://pypi.org/project/deepcoder/0.2.6/).

### Setup API key

Choose **one** of:
- Export env variable (you can add this to .bashrc so that you don't have to do it each time you start the terminal)
    - `export OPENAI_API_KEY=[your api key]`
- .env file:
    - Create a copy of `.env.template` named `.env`
    - Add your OPENAI_API_KEY in .env
- Custom model:
    - See [docs](https://deepcoder.readthedocs.io/en/latest/open_models.html), supports local model, azure, etc.

Check the [Windows README](./WINDOWS_README.md) for Windows usage.

**Other ways to run:**
- Use Docker ([instructions](docker/README.md))
- Do everything in your browser:
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/deepcode-ai/deepcoder/codespaces)

### Create new code (default usage)
- Create an empty folder for your project anywhere on your computer
- Create a file called `prompt` (no extension) inside your new folder and fill it with instructions
- Run `gpte <project_dir>` with a relative path to your folder
  - For example: `gpte projects/my-new-project` from the deepcoder directory root with your new folder in `projects/`

### Improve existing code
- Locate a folder with code which you want to improve anywhere on your computer
- Create a file called `prompt` (no extension) inside your new folder and fill it with instructions for how you want to improve the code
- Run `gpte <project_dir> -i` with a relative path to your folder
  - For example: `gpte projects/my-old-project -i` from the deepcoder directory root with your folder in `projects/`

### Benchmark custom agents
- deepcoder installs the binary 'bench', which gives you a simple interface for benchmarking your own agent implementations against popular public datasets.
- The easiest way to get started with benchmarking is by checking out the [template](https://github.com/deepcoder-org/gpte-bench-template) repo, which contains detailed instructions and an agent template.
- Currently supported benchmark:
  - [APPS](https://github.com/hendrycks/apps)
  - [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)

The community has started work with different benchmarking initiatives, as described in [this Loom](https://www.loom.com/share/206805143fbb4302b5455a5329eaab17?sid=f689608f-8e49-44f7-b55f-4c81e9dc93e6) video.

### Research
Some of our community members have worked on different research briefs that could be taken further. See [this document](https://docs.google.com/document/d/1qmOj2DvdPc6syIAm8iISZFpfik26BYw7ZziD5c-9G0E/edit?usp=sharing) if you are interested.

## Terms
By running deepcoder, you agree to our [terms](https://github.com/deepcode-ai/deepcoder/blob/main/TERMS_OF_USE.md).


## Relation to deepcoder.app (Deep Coder)
[deepcoder.app](https://deepcoder.app/) is a commercial project for the automatic generation of web apps.
It features a UI for non-technical users connected to a git-controlled codebase.
The deepcoder.app team is actively supporting the open source community.


## Features

### Pre Prompts
You can specify the "identity" of the AI agent by overriding the `preprompts` folder with your own version of the `preprompts`. You can do so via the `--use-custom-preprompts` argument.

Editing the `preprompts` is how you make the agent remember things between projects.

### Vision

By default, deepcoder expects text input via a `prompt` file. It can also accept image inputs for vision-capable models. This can be useful for adding UX or architecture diagrams as additional context for Deep Coder. You can do this by specifying an image directory with the `—-image_directory` flag and setting a vision-capable model in the second CLI argument.

E.g. `gpte projects/example-vision gpt-4-vision-preview --prompt_file prompt/text --image_directory prompt/images -i`

### Open source, local and alternative models

By default, deepcoder supports OpenAI Models via the OpenAI API or Azure OpenAI API, as well as Anthropic models.

With a little extra setup, you can also run with open source models like WizardCoder. See the [documentation](https://deepcoder.readthedocs.io/en/latest/open_models.html) for example instructions.

## Mission

The deepcoder community mission is to **maintain tools that coding agent builders can use and facilitate collaboration in the open source community**.

If you are interested in contributing to this, we are interested in having you.

If you want to see our broader ambitions, check out the [roadmap](https://github.com/deepcode-ai/deepcoder/blob/main/ROADMAP.md), and join
[discord](https://discord.gg/8tcDQ89Ej2)
to learn how you can [contribute](.github/CONTRIBUTING.md) to it.

deepcoder is [governed](https://github.com/deepcode-ai/deepcoder/blob/main/GOVERNANCE.md) by a board of long-term contributors. If you contribute routinely and have an interest in shaping the future of deepcoder, you will be considered for the board.
