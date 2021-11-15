from distutils.core import setup

setup(
    name="torchir",
    packages=["torchir"],  # Chose the same as "name"
    version="0.1",
    license="MIT",
    description="Pytorch Image Registration Library",
    author="Bob D. de Vos",  # Type in your name
    author_email="b.d.devos@amsterdamumc.nl",
    url="https://github.com/BDdeVos/TorchIR",
    download_url="https://github.com/BDdeVos/TorchIR/archive/refs/tags/version_01.tar.gz",  # I explain this later on
    keywords=[
        "PyTorch",
        "Deep Learning",
        "Image Registration",
        "Affine Registration",
        "Deformable Registration",
    ],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        "torch",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
