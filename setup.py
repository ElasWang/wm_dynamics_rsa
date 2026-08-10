from setuptools import setup, find_packages

setup(
    name='wm_dynamics_rsa',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'mne>=1.7.0',
        'numpy>=1.26.0',
        'torch>=2.0.0',
    ],
    python_requires='>=3.10',
)