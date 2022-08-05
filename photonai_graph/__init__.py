import os
from datetime import datetime
from photonai.photonlogger import logger

__version__ = '0.1.0'

current_path = os.path.dirname(os.path.abspath(__file__))
registered_file = os.path.join(current_path, "registered")
logger.info("Checking Graph Module Registration")
if not os.path.isfile(registered_file):  # pragma: no cover
    logger.info("Registering Graph Module")
    from photonai.base import PhotonRegistry
    reg = PhotonRegistry()
    reg.add_module(os.path.join(current_path, "photonai_graph.json"))
    with open(os.path.join(registered_file), "w") as f:
        f.write(str(datetime.now()))
