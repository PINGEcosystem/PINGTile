'''
Shared runtime preflight helpers for mapper packages.
'''

#########
# Imports
import os
import sys


#=======================================================================
def prepare_windows_torch_runtime(preload_torch: bool = True):
    '''
    Configure Windows torch runtime paths and optionally preload torch.
    '''
    if os.name != 'nt':
        return

    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    torch_lib_dir = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.isdir(torch_lib_dir):
        os.environ['PATH'] = torch_lib_dir + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(torch_lib_dir)
        except Exception:
            pass

    if preload_torch:
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            # Caller may handle torch import requirements later.
            pass


#=======================================================================
def prepare_windows_geo_runtime():
    '''
    Configure GDAL/PROJ environment variables on Windows.
    '''
    if os.name != 'nt':
        return

    share_dir = os.path.join(sys.prefix, 'Library', 'share')
    gdal_data_dir = os.path.join(share_dir, 'gdal')
    proj_data_dir = os.path.join(share_dir, 'proj')

    if os.path.isdir(gdal_data_dir):
        os.environ.setdefault('GDAL_DATA', gdal_data_dir)

    if os.path.isdir(proj_data_dir):
        os.environ.setdefault('PROJ_LIB', proj_data_dir)


#=======================================================================
def prepare_windows_mapper_runtime(preload_torch: bool = True):
    '''
    Configure both torch and geospatial runtime settings for mapper apps.
    '''
    prepare_windows_torch_runtime(preload_torch=preload_torch)
    prepare_windows_geo_runtime()
