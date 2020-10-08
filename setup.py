try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

__version__ = '0.1.1'

setup(
    name='photonai-graph',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description="""
PHOTON Graph
PHOTON Graph is a submodule of the PHOTON package, which, using the PHOTON API, allows the user to utilize graph based machine learning methods
within the PHOTON framework. The module provides a range of graph machine learning algorithms, along with methods for constructing graphs.
""",
    author='PHOTON Team',
    author_email='v_hols01@uni-muenster.de',
    url='https://github.com/wwu-mmll/photonai_graph.git',
    download_url='https://github.com/wwu-mmll/photonai_graph/archive/' + __version__ + '.tar.gz',
    keywords=['machine learning', 'deep learning', 'graph convolutional neural networks', 'graphs'],
    classifiers=[],
    install_requires=['photonai>=2.0.0',
                      'networkx>=2.4',
                      'pydot>=1.4.1',
                      'gem @ git+https://github.com/palash1992/GEM.git@5da663255da28c433c42b296c8ceed7163f2d509',
                      'grakel @ git+https://github.com/ysig/GraKeL.git@cfd14e0543075308d201327ac778a48643f81095',
                      'torch>=1.6.0',
                      'dgl>=0.5.2',
                      'pandas>=1.1.2',
                      'numpy>=1.12.2',
                      'scipy>=0.19.1']
)
