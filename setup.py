import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mesostat-alfomi", # Replace with your own username
    version="0.0.1",
    author="Aleksejs Fomins",
    author_email="aleksejs.fomins@uzh.ch",
    description="Statistics package for analysis of mesoscopic neuronal data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aleksejs-fomins/mesostat-dev",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
