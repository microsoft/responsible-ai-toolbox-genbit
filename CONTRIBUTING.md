# Contributing to the Gender Bias Tool

## Working locally
To get genbit to work locally please run `pip install -r requirements.txt` to install all requirements.

## Testing and linting
Please add appropriate tests for to every piece of code you contribute. To run all tests for this repository, run `python -m unittest` with the virtual environment enabled. 

### Pushing code
Please run the auto linter followed by all unittests before pushing any code:
```
autopep8 --in-place --recursive --aggressive .
python -m unittest
```
Make sure that there are no pylint issues by running `pylint genbit`

### Adding 3rd party python package dependencies
When adding a new 3rd party dependency it is important to make sure that the licenses of these packages allow us to use the packages for our usage purposes. The easiest way to do this is to add the package and version to [requirements.txt](requirements.txt). By doing this, they will be included in the azure-pipeline build. You can then go to the [Component Governance section in ADO](https://office.visualstudio.com/GSX/_componentGovernance/GenderBiasTool) to validate whether new packages have triggerd any alerts, if so, please review these alerts. 

## Sharing GenBiT

### Python Package
The code can be cloned or seen from here: [GenBiT source code](https://aka.ms/genbit)

### Uploading the package to azure artifacts with twine
The azure artifacts location for GenBiT can be found [here](https://office.visualstudio.com/GSX/_packaging?_a=settings&feed=GenderBias)
1. Install twine and azure artifacts: `pip install artifacts-keyring --pre` `pip install twine wheel`
2. To build the package, run `py setup.py sdist bdist_wheel`
3. To upload to twine use `twine upload dist/*  --repository-url https://office.pkgs.visualstudio.com/GSX/_packaging/GenderBias/pypi/upload/`

