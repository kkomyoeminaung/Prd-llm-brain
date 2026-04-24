from setuptools import setup, find_packages

setup(
    name="prd-llm",
    version="2.0.0",
    author="MyoMinAung Research Team",
    description="Human-Like Brain Architecture for AGI",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "prd-llm=prd_llm.cli:main",
        ],
    },
)
