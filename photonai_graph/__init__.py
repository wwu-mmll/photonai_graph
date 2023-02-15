import os
from photonai.base import PhotonRegistry
from photonai.photonlogger import logger

from .version import __version__


def do_register(current_path, registered_file):
    reg = PhotonRegistry()
    reg.add_module(os.path.join(current_path, "photonai_graph.json"))
    with open(os.path.join(registered_file), "w") as f:
        f.write(str(__version__))


def register():
    current_path = os.path.dirname(os.path.abspath(__file__))
    registered_file = os.path.join(current_path, "registered")
    logger.info("Checking PHOTONAI Graph Module Registration")
    if not os.path.isfile(registered_file):  # pragma: no cover
        logger.info("Registering Graph Module")
        do_register(current_path=current_path, registered_file=registered_file)
    else:
        with open(os.path.join(registered_file), "r") as f:
            if f.read() == __version__:
                logger.info("Current version already registered")
            else:
                logger.info("Updating Graph Module")
                do_register(current_path=current_path, registered_file=registered_file)


register()
