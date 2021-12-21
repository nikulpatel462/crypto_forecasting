from setuptools import setup

setup(
    name="kaggle-crypto-lib",
    version="0.0",
    description="Kaggle G-Research Crypto forecasting library",
    author="NP",
    install_requires=[
        "wheel",
        "numpy",
        "pandas",
        "lightgbm",
        "scikit-learn>1.0",
        "tensorflow",
    ],  # external packages as dependencies
)
