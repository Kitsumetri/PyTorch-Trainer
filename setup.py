from setuptools import setup, find_packages

setup(
    name="torch-trainer",
    version="0.1.0",
    author="Kitsumetri",
    author_email="-",
    description="A library to simplify training models in PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Kitsumetri/PyTorchTrainer",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorboard",
        "torch",
        "torchaudio",
        "torchvision",
        "tqdm",
        "numpy",
        "ruff",
    ],
    python_requires=">=3.10",
)
