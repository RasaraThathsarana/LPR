"""Root training entry point."""

import runpy

if __name__ == '__main__':
    runpy.run_module('training.train', run_name='__main__')
