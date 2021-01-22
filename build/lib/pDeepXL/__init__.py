import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data_abs_path(rel_to_root_path):
    return os.path.join(_ROOT, 'data', rel_to_root_path)
    