"""Root module of your package"""
import os

topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_version():
  with open(os.path.join(topdir, "version.txt")) as f:
    return f.read()

__version__ = get_version()
