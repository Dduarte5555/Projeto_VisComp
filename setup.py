from setuptools import setup, find_packages

setup(
    name="mri_cv_tools",
    version="0.1.0",
    description="Ferramentas para processamento de imagens médicas, treinamento de modelos para exames de MR",
    author="Equipe LeMON",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'segmentation-models-pytorch',
        'torchmetrics',
        'nibabel',
        'torch',
        'numpy',
        'albumentations',
        'pathlib',
        'typing',
        'classification-models-3D',
        'efficientnet-3D',
        'segmentation-models-3D',
        'scikit-image',
        'matplotlib',
        'scikit-learn',
        'patchify',
        'tensorflow',
        'pandas',
    ],
    extras_require={
        "dev": ["pytest>=7.0"]
    },
    python_requires=">=3.11",
)