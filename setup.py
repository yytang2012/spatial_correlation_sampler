from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spatial-correlation-sampler-pytorch",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Pure PyTorch implementation of Spatial Correlation Sampler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spatial-correlation-sampler-pytorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.4.0",
        "numpy>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark",
            "black",
            "flake8",
        ],
    },
)