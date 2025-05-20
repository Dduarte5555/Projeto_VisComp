from setuptools import setup, find_packages

setup(
    name="LeMON",
    version="0.1.0",
    description="Ferramentas para pré-processamento de imagens médicas: DICOM→NIfTI, máscaras e cálculo de distância",
    author="Equipe LeMON",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydicom>=2.0.0",
        "nibabel>=4.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0"
    ],
    extras_require={
        "dev": ["pytest>=7.0"]
    },
    python_requires=">=3.8",
)