import setuptools

setuptools.setup(
    name='daami2i',
    version=eval(open('daami2i/_version.py').read().strip().split('=')[1]),
    author='Rishi Dey Chowdhury',
    license='MIT',
    url='https://github.com/RishiDarkDevil/daam-i2i',
    author_email='rishi8001100192@gmail.com',
    description='DAAM-i2i: Interpreting Stable Diffusion Using Self Attention.',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'daam = daam.run.generate:main',
            'daam-demo = daam.run.demo:main',
        ]
    }
)
