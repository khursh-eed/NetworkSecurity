from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    # this function will return list of requirements

    requirement_list: List[str]=[]

    try: 
        with open ('requirements.txt','r') as file:
            # read lines from file
            lines =file.readlines()
            for line in lines:
                requirements = line.strip()
                # ignore empty line & -e.
                if requirements and requirements != '-e .':
                    requirement_list.append(requirements)
    except FileNotFoundError:
        print("requirements.txt not found")
    return requirement_list 

# print(get_requirements())

setup(
    name = "NetworkSecurity",
    version="0.01",
    author="Khursheed",
    author_email="kfm.hyd2005@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements()
)
