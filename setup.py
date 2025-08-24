from setuptools import setup, find_packages

setup(
    name="ml-model-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "Flask==2.3.3",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "joblib==1.3.2",
        "gunicorn==21.2.0",
    ],
    python_requires=">=3.8",
)