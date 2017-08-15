from setuptools import setup, find_packages

setup(
    name='reactionrnn',
    packages=['reactionrnn'],  # this must be the same as the name above
    version='0.1',
    description='Pretrained character-based neural network for ' \
    'predicting the reaction to given text(s).',
    author='Max Woolf',
    author_email='max@minimaxir.com',
    url='https://github.com/minimaxir/reactionrnn',
    keywords=['deep learning', 'tensorflow', 'keras', 'sentiment analysis'],
    classifiers=[],
    license='MIT',
    include_package_data=True,
    install_requires=['tensorflow', 'keras', 'h5py']
)
