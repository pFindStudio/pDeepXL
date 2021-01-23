import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# https://stackoverflow.com/a/36693250
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('pDeepXL/examples')

setuptools.setup(
    name="pDeepXL",
    version="1.1.4",
    author="Zhenlin Chen",
    author_email="chenzhenlin@ict.ac.cn",
    description="MS/MS spectrum prediction for cross-linked peptide pairs by deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pFindStudio/pDeepXL",
    packages=setuptools.find_packages(),
    package_data={'pDeepXL': ['configs/*','pt/*'] + extra_files},
    entry_points={
    'console_scripts': [
        'pDeepXL_predict_save_batch=pDeepXL.console_wapper:predict_save_batch',
        'pDeepXL_predict_save_plot_batch=pDeepXL.console_wapper:predict_save_plot_batch',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'tqdm',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy',
        'torch',
        'torchvision',
    ],
)