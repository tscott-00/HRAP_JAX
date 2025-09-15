# pip instasll --force-reinstall git+https://github.com/tscott-00/HRAP_JAX@webview

import os
import sys
import time
from pathlib import Path
import pickle as pkl
from importlib.resources import files as imp_files
from importlib.metadata import version

import scipy
import numpy as np

import dearpygui.dearpygui as dpg

from jax.scipy.interpolate import RegularGridInterpolator

import hrap.core as core
import hrap.chem as chem
import hrap.fluid as fluid
import hrap.units as units
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.units   import _in, _ft
import webview

hrap_version = version('hrap')

# Globals
hrap_root = None
window = None

def click_handler(e):
    print(e['target'])

def input_handler(e):
    print(e['target']['value'])

def testing(window):
    # result = window.create_confirmation_dialog('Notice', 'Your credit card has been stolen')
    # if result:
        # print('User clicked OK')
    # else:
        # print('User clicked Cancel')
    
    # button = window.dom.get_element('#button')
    # button.events.click += click_handler

    # input = window.dom.get_element('#input')
    # input.events.input += input_handler
    pass

# def load_hrap_html(local_path):
    # with open(hrap_root/'gui'/local_path, mode='rt') as file:
        # html = file.read()
    # return html

# TODO:
# drag and drop? https://pywebview.flowrl.com/examples/drag_drop.html
# on resized, on move https://pywebview.flowrl.com/examples/events.html resize https://pywebview.flowrl.com/examples/resize.html move https://pywebview.flowrl.com/examples/move_window.html
# full screen https://pywebview.flowrl.com/examples/toggle_fullscreen.html
# glass theme? walter https://pywebview.flowrl.com/examples/transparent.html

def main():
    global hrap_root, window
    
    # Get the HRAP install root
    hrap_root = Path(imp_files('hrap'))
    
    # window = webview.create_window('HRAP', 'https://pywebview.flowrl.com/')
    
    # landing_html = load_hrap_page
    # window = webview.create_window('HRAP', 'https://www.ssta.club/')
    # print(load_hrap_html('index.html'))
    # window = webview.create_window('HRAP', html=load_hrap_html('index.html'))
    # window = webview.create_window('HRAP', html=load_hrap_html('frontend/dist/index.html'))
    window = webview.create_window('HRAP', str(hrap_root/'gui'/'frontend'/'dist'/'index.html'))
    # webview.settings['REMOTE_DEBUGGING_PORT'] = 4001 # can go to via chrome://inspect but is slow
    # HTTP server
    # webview.start(testing, window)
    webview.start(testing, window, http_server=True, debug=True)

if __name__ == '__main__': main()
    
