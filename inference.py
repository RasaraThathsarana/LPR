"""Root inference entry point."""

import runpy

if __name__ == '__main__':
    runpy.run_path('inference/inference.py', run_name='__main__')
