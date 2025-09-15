from setuptools import setup, find_packages

setup(
    name="labgraphcast",
    version="0.0.1",
    description="Scalable GNN forecasting for lab occupancy with multilevel graph compression (CMG).",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "streamlit",
        "scikit-learn",
        "statsmodels",
        # "torch", "torch-geometric", etc. can be added later
    ],
    python_requires=">=3.8",
)
