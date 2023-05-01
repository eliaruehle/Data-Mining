# How to Work with Poetry

## 1. Installing Dependencies

To install the dependencies for your project, you can use the `install` command:

```
$ poetry install
```

This will read your `pyproject.toml` file and install all the required dependencies.

## Adding Dependencies (Optional)

To add a dependency to your project, you can use the `add` command. This will add the specified package to your project's dependencies and update your `pyproject.toml` file:

```
$ poetry add package-name
```

You can also specify a specific version of the package by appending `@version` to the package name.

## Managing Virtual Environments

Poetry creates and manages virtual environments for your projects automatically. To activate the virtual environment for your project, you can use the `shell` command:

```
$ poetry shell
```

This will activate the virtual environment and allow you to work within it. To deactivate the virtual environment, you can use the `exit` command.

## Running Scripts

You can define scripts in your `pyproject.toml` file and run them using the `run` command:

```
$ poetry run script-name
```

This will run the specified script within your project's virtual environment.
