try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

__version__ = '0.1.1'

setup(
    name='photonai_graph',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description="""
PHOTON Graph
PHOTON Graph is a submodule of the PHOTON package, which, using the PHOTON API, allows the user to utilize graph based machine learning methods
within the PHOTON framework. The module provides a range of graph machine learning algorithms, along with methods for constructing graphs.
""",
    author='PHOTON Team',
    author_email='hahnt@uni-muenster.de',
    url='https://github.com/wwu-mmll/photonai_graph.git',
    download_url='https://github.com/wwu-mmll/photonai_graph/archive/' + __version__ + '.tar.gz',
    keywords=['machine learning', 'deep learning', 'graph convolutional neural networks', 'graphs'],
    classifiers=[],
    install_requires=['photonai>2.2.0',
                      'networkx>=2.4',
                      'pydot>=1.4.1',
                      'nxt-gem',
                      'pandas>=1.1.2',
                      'numpy>=1.12.2',
                      'scipy>=0.19.1',
                      'tqdm']
)
