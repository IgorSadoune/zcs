from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="zcsearch",
    version="0.1.3",
    author="Igor Sadoune",
    author_email="igor.sadoune@pm.me",
    description="Zero-Cost Neural Architecture Search for MLPs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IgorSadoune/zcsearch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.7.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "csv": ["pandas>=1.0.0"],
    },
    entry_points={
        'console_scripts': [
            'zcsearch=zcsearch.cli:main',
        ],
    },
)