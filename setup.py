import os
from setuptools import setup

def get_requirements():
    requirements = [line.strip() for line in open('requirements.txt')]
    return requirements

def get_version():
    versionFile = "genbit/version.py"
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, versionFile), 'r') as reader:
        lines = reader.read()

    for line in lines.splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")

setup = setup(
    name='genbit',
    version=get_version(),
    packages=["genbit", "genbit.gendered-word-lists", "genbit.tests"],
    include_package_data=True,
    package_dir={'genbit.gendered-word-lists': os.path.join('genbit', 'gendered-word-lists'), 'genbit.tests': 'tests'},
    license='MIT License',
    url='https://github.com/microsoft/responsibleaitoolbox-genbit',
    install_requires=get_requirements(),
    author="Microsoft",
    author_email='raiwidgets-maintain@microsoft.com',
    long_description="Main goal of the Gender Bias Tool (GenBiT) is to analyze corpora and compute metrics that give insights into the gender bias present in a corpus.\n \nFor more information please visithttps://github.com/microsoft/responsibleaitoolbox-genbit. "
)
