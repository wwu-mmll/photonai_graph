try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

__version__ = '0.0.1'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='photonai_graph',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="PHOTON Graph - Graph machine learning with photonai.",
    author='PHOTON Team',
    author_email='hahnt@uni-muenster.de',
    url='https://github.com/wwu-mmll/photonai_graph.git',
    project_urls={
        "Source Code": "https://github.com/wwu-mmll/photonai_graph",
        "Documentation": "https://wwu-mmll.github.io/photonai_graph/",
        "Bug Tracker": "https://github.com/wwu-mmll/photonai_graph/issues",
    },
    download_url='https://pypi.org/project/photonai-graph/#files',
    keywords=['machine learning', 'deep learning', 'graph convolutional neural networks', 'graphs'],
    classifiers=["License :: OSI Approved :: MIT License",
                 "Topic :: Software Development :: Libraries :: Python Modules",
                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                 "Intended Audience :: Science/Research"],
    install_requires=['photonai>2.2.0',
                      'networkx>=2.4',
                      'pydot>=1.4.1',
                      'nxt-gem',
                      'pandas>=1.1.2',
                      'numpy>=1.12.2',
                      'scipy>=0.19.1',
                      'tqdm']
)
