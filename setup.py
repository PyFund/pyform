import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

install_reqs = ["pandas>=1.0.0"]

setuptools.setup(
    name="pyform",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Shawn Lin",
    author_email="shawn.lin@gatech.edu",
    description="An event-driven algorithmic trading framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shawnlinxl/pyform",
    packages=setuptools.find_packages(),
    install_requires=install_reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
