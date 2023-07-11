from setuptools import setup, find_packages

setup(
    name="cell_type_mapper",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)
