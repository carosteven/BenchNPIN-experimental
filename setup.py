from setuptools import setup, find_packages

setup(
    name="namo_envs",
    version="0.0.1",
    install_requires=["gymnasium"],
    packages=find_packages(), 
    include_package_data=True,
    package_data={
        'namo_envs.env_configs':['*.yaml']
    }
)