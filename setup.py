from setuptools import setup, find_packages

setup(
   name='difai',
   version='1.0',
   author='Markus Klar, Samuel Younger',
   author_email='markus.klar@glasgow.ac.uk, Samuel.Younger24@imperial.ac.uk',
   packages=['difai'],
   package_data={'': ['**']},
   url='https://github.com/youngers2006/difai-base.git',
   license='LICENSE',
   python_requires='>=3.11',
   install_requires=[
       "numpy", "optax", "tqdm", "pyyaml", "matplotlib"
   ],
   extras_require={
    "gpu": ["jax[cuda13]"],
    "tpu": ["jax[tpu]"],
    "cpu": ["jax"],
    },
)