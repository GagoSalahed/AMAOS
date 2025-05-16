from setuptools import setup, find_packages

setup(
    name="amaos",
    version="0.1.0",
    description="Advanced Multi-Agent Orchestration System",
    author="AMAOS Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.5.0",
        "asyncio>=3.4.3",
        "typing-extensions>=4.7.1",
        "prometheus-client>=0.17.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.1",
            "pytest-timeout>=2.1.0",
            "pytest-mock>=3.11.1",
            "mypy>=1.5.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
)
