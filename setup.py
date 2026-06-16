from setuptools import find_packages, setup

setup(
    name="model-taxonomy",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "datasets>=2.20.0",
        "sentence-transformers>=3.0.0",
        "scikit-learn>=1.5.0",
        "submitit>=1.5.1",
        "joblib>=1.4.0",
        "huggingface-hub>=0.25.0",
        "filelock>=3.13.0",
        "networkx>=3.0.0",
        "safetensors>=0.4.0",
        "peft>=0.12.0",
        "trl>=0.11.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "umap": ["umap-learn>=0.5.0"],
    },
)
