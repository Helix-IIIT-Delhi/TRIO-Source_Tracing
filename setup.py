from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trio-source-tracing",
    version="1.0.0",
    author="Girish, Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Swarup Ranjan Behera, Rajesh Sharma",
    description="TRIO: Source Tracing of Synthetic Speech Systems via Paralinguistic Representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helix-IIIT-Delhi/TRIO-Source_Tracing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
