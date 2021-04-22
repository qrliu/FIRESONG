import setuptools

long_message = 'FIRESONG: the FIRst Extragalactic Simulation Of Neutrinos and Gamma-rays'
version = "1.6.0"

setuptools.setup(
    name="firesong", 
    version=version,
    author="Tung, C.F. et al.",
    author_email="",
    description="Code for simulationg populations of neutrino sources",
    long_description=long_message,
    #long_description_content_type="text/markdown",
    url="https://github.com/icecube/FIRESONG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires='>=3.1',
)
