"""Vendored SegFace model code.

Source: https://github.com/Kartik-3004/SegFace (MIT licence, AAAI 2025).
Files segface_celeb.py, transformer.py and utils_models.py are copied verbatim
apart from the import rewrite below -- the originals do sys.path surgery to find
each other, which does not survive being imported as a package.
"""
