# Contributing to the Gender Bias Tool

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

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
When adding a new 3rd party dependency it is important to make sure that the licenses of these packages are compatible with the MIT License and adhere to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). The easiest way to do this is to add the package and version to [requirements.txt](requirements.txt). By doing this, they will be included in the azure-pipeline build. You can then go to the [Component Governance section in ADO](https://office.visualstudio.com/GSX/_componentGovernance/GenderBiasTool) to validate whether new packages have triggered any alerts, if so, please review these alerts. 

## Sharing GenBiT

### Python Package
The code can be cloned or seen from here: [GenBiT source code](https://github.com/microsoft/responsibleaitoolbox-genbit)


