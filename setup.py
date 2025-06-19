from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of requirements.
    It removes any '-e .' entry which is used for editable installs.
    """
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]  # Remove newline characters
    
    # Remove any '-e .' entry
    # requirements = [req.strip() for req in requirements if req.strip() != "-e ."]

    if "-e ." in requirements:
        requirements.remove("-e .")
    
    return requirements 

setup(
    name="Ml_project",
    version="0.0.1",
    author="sahdev",
    author_email="sahdevpithiy46@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)