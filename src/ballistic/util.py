import os
import pathlib

def project_path():
    return pathlib.Path(__file__).parent.parent.parent.absolute()

def resource(rel_path : str):
    base_path = os.path.join(project_path(), 'res')
    return os.path.abspath(os.path.join(base_path, rel_path))

def all_paths(base_root : str) -> list[str]:
    paths = []
    for root,_,files in os.walk(base_root, topdown=True):
        prefix = (
            ""
            if root == base_root else
	    root[len(base_root) + 1:] + '/'
        )
        for f in files:
            paths.append(prefix + f)
    return paths


# def write(dirpath : str, fname : str, code : str, append : bool = False):
#     if not os.path.exists(dirpath):
#         os.makedirs(dirpath)

#     fpath = os.path.join(dirpath, f"{fname}")

#     with open(fpath, 'a' if append else 'w') as f:
#         # logging.info(f"Writing file: {fpath}")
#         f.write(code)

def read(fpath : str):
    with open(fpath) as f:
        return f.read()