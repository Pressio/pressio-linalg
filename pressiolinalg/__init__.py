"""Root module of your package"""
import os

topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(topdir, "version.txt")) as f:
    __version__ = f.read().strip()
