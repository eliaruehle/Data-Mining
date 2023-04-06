# Code Style and Linting Conventions

In order to maintain a consistent and high-quality codebase, we follow the guidelines and conventions outlined below.

## Indentation

All indentation should be done using 4 spaces. One tab character should be equal to 4 spaces.

## Naming Conventions

- Snake case should be used for all variable and function names.
- Camel case should be used for all class names.
- Constants should be written in all caps with underscores separating words.
- Enum members should be written in all caps with underscores separating words.

```
my_variable = 42
```

```
def my_function(argument1, argument2): # ...
```

```
class MyClass: # ...
```

```
MY_CONSTANT = 42
```

## Linting

We use [Flake8](https://flake8.pycqa.org/en/latest/) and [Black](https://github.com/psf/black) for linting and formatting our code.

### Flake8 Configuration

Our Flake8 configuration is located in the `pyproject.toml` file and includes the following settings:

We ignore the `E203` and `W503` errors because they conflict with Black's formatting. The `max-line-length` is set to 88 characters. We use the `select` option to choose the error codes to check for, and `exclude` option to exclude some directories from linting.

### Black Configuration

Our Black configuration is also located in the `pyproject.toml` file and includes the following settings:

[tool.black]
line-length = 88
include = '.pyi?$'
exclude = '''
/(
.eggs/ |
.git/ |
.hg/ |
.mypy_cache/ |
.nox/ |
.tox/ |
.venv/ |
\_build/ |
buck-out/ |
build/ |
dist/ |
node_modules/ |
out/ |
venv/ |
yarn-error.log
)/
'''

We set the `line-length` to 88 characters and use the `include` and `exclude` options to specify which files to format and which directories to exclude.

### MyPy Configuration (Optional)

Optionally, we use [MyPy](http://mypy-lang.org/) for static type checking. Our MyPy configuration is located in the `pyproject.toml` file and includes the following settings:

[mypy]
plugins = sqlmypy
python_version = 3.9
ignore_missing_imports = True
warn_return_any = True
warn_unused_configs = Tru

We use the `plugins` option to specify additional plugins (in this case, `sqlmypy`). The `python_version` option specifies the version of Python we're using, and `ignore_missing_imports`, `warn_return_any`, and `warn_unused_configs` are options for controlling which errors to report.
