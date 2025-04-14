from setuptools import setup, find_packages

try:
    with open("requirements.txt") as f:
        required_packages = f.read().splitlines()
except FileNotFoundError:
    required_packages = []
    
setup(
    
    name="Food-delivery-time-predictions",
    version="0.0.1",
    author="Vaibhav Rai",
    author_email="vairaibhav922@gmail.com",
    description= " End to end mlops project",
    packages=find_packages(),
    install_requires=required_packages,
)