import os
from setuptools import setup, find_packages

setup(
    name="retrieval-methods",
    version="0.1.0",
    author="agentmaddi",
    description="RAG system with probability-stasis filtering for stable, reliable results",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "chromadb>=1.1.1",
        "sentence-transformers>=5.1.1",
        "numpy>=2.0.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
)
