# # pip instasll --force-reinstall git+https://github.com/tscott-00/HRAP_JAX@webview

import hrap

import os
import sys
import time
import argparse
from pathlib import Path
import pickle as pkl
from importlib.resources import files as imp_files
from importlib.metadata import version

import scipy
import numpy as np

# import dearpygui.dearpygui as dpg
# both raw webview and nicegui take around 5s to start up...
# from bottle import Bottle, route, run, static_file
# import webview
from nicegui import app, ui, native, binding

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

hrap_version = version('hrap')
hrap_root = Path(imp_files('hrap')) # HRAP install root
html_root = hrap_root/'gui'/'frontend'/'dist' # root for files used in HTML GUI
# app = Bottle()

# Globals to be set later
window = None

# def click_handler(e):
    # print(e['target'])

# def input_handler(e):
    # print(e['target']['value'])

# def testing(window):
    # # result = window.create_confirmation_dialog('Notice', 'Your credit card has been stolen')
    # # if result:
        # # print('User clicked OK')
    # # else:
        # # print('User clicked Cancel')
    
    # # button = window.dom.get_element('#button')
    # # button.events.click += click_handler

    # # input = window.dom.get_element('#input')
    # # input.events.input += input_handler
    # pass

# def load_hrap_html(local_path):
    # with open(hrap_root/'gui'/'frontend'/'dist'/local_path, mode='rt') as file:
        # html = file.read()
    # return html

# TODO:
# drag and drop? https://pywebview.flowrl.com/examples/drag_drop.html
# on resized, on move https://pywebview.flowrl.com/examples/events.html resize https://pywebview.flowrl.com/examples/resize.html move https://pywebview.flowrl.com/examples/move_window.html
# full screen https://pywebview.flowrl.com/examples/toggle_fullscreen.html
# glass theme? walter https://pywebview.flowrl.com/examples/transparent.html

# @app.route('/')
# def home():
    # with open(html_root/'index.html', mode='rt') as file:
        # html = file.read()
    # return html

# # Serve static files
# @app.route('/assets/<filename:path>')
# def serve_static(filename):
    # print('serving', filename)
    # return static_file(filename, root=html_root/'assets')

from nicegui.element import Element

# @binding.bindable_dataclass
class TankSlider(Element, component='Tank.vue'):
    # value: float = 0.4
# class TankSlider(Element, component='OnOff.vue'):
    # def __init__(self, title: str, *, on_change: Optional[Callable] = None) -> None:
    def __init__(self, title: str, *, a=0) -> None:
        super().__init__()
        self._props['title'] = title
        # # self.on('change', on_change)

    # # def reset(self) -> None:
        # # self.run_method('reset')

# TODO: any way to autosave on reload?
# Direct call is if ran via "python .\hrap\gui\main.py" instead of "hrap," which enables development features such as auto reload on file change
def main(is_direct_call=False):
    global window #, __name__
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='Open debug, i.e. inspect-element, window', action='store_true')
    # parser.add_argument('--path',  help='Data and plots are written within ./results/nnnfgp/[path]', type=str, default='')
    args = parser.parse_args()
        
    # landing_html = load_hrap_page
    # window = webview.create_window('HRAP', str(hrap_root/'gui'/'frontend'/'dist'/'index.html'))
    
    with ui.row(align_items='center'):
        

        tank_slider = TankSlider('test')
        # tank_slider.bind_value(tank_value, 'value')
        # tank_value = ui.number('Tank Value', value=0.5).props('color=blue')#.bind_value(tank_slider, 'value')
        ui.button('Jiggle', on_click=lambda: tank_slider.run_method('jiggle')).props('outline')
    
    ui.run(reload=is_direct_call, uvicorn_reload_includes='*.py,*.js,*.vue')
    # ui.run(native=True, reload=is_direct_call, port=native.find_open_port())

if __name__ in {'__main__', '__mp_main__'}: main(is_direct_call=True)
# if __name__ == '__main__': main(is_direct_call=True)
