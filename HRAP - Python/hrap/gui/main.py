# Copyright 2026 The HRAP Authors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Doesn't work, needs to be python subfolder...
# pip install --force-reinstall git+https://github.com/tscott-00/HRAP_JAX@webview

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
from hrap.units   import _in, _ft, unit_conversions, inv_unit_conversions

hrap_version = version('hrap')
hrap_root = Path(imp_files('hrap')) # HRAP install root
# html_root = hrap_root/'gui'/'frontend'/'dist' # root for files used in HTML GUI
# app = Bottle()

# Globals to be set later
webview = None # Package, only imported when run in native winow
window = None

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

# HTML code to nicely display units
def get_unit_html_name(unit_type, unit, _names={
    'area': {f'{base}2': f'{base}<sup>2</sup>' for base in ['mm', 'in']},
    'temperature': {'C': '&deg;C', 'F': '&deg;F'},
}):
    _names.get(unit_type,{}).get(unit,unit)

# @binding.bindable_dataclass
class TankSlider(Element, component='components/Tank.vue'):
    # value: float = 0.4
# class TankSlider(Element, component='OnOff.vue'):
    # def __init__(self, title: str, *, on_change: Optional[Callable] = None) -> None:
    def __init__(self, title: str, *, a=0) -> None:
        super().__init__()
        self._props['title'] = title
        # # self.on('change', on_change)

    # # def reset(self) -> None:
        # # self.run_method('reset')


def landing():
    # ui.label('Welcome to HRAP! Choose an analysis mode below.')
    # ui.separator()
    ui.label('Analysis').classes('text-h6')
    ui.separator()
    with ui.row(align_items='center'):
        for title, desc, mode, thumb_path in [
            ('Hybrid Engine', 'Self-pressurizing liquid oxidizer, solid fuel', 'hybrid', hrap_root/'gui'/'thumbnails'/'hybrid.jpg'),
            ('Liquid Engine', 'Half-Cat-style liquid oxidizer and fuel', 'liquid', hrap_root/'gui'/'thumbnails'/'liquid.jpg'),
            ('Solid Motor',   'Solid oxidizer and fuel', 'solid', hrap_root/'gui'/'thumbnails'/'placeholder.jpg'),
            ('Chemistry',     'Analyze combustion without engine details', 'chem', hrap_root/'gui'/'thumbnails'/'chemistry.jpg'),
        ]:
            with ui.card().tight().classes('w-80'):
                with ui.image(thumb_path):
                    with ui.element('div').classes('absolute-bottom text-white p-2'):
                        ui.label(title).classes('text-h6')
                        ui.label(desc).classes('text-subtitle2')
                with ui.card_section():
                    ui.button('New',    on_click=lambda mode=mode: ui.navigate.to('/'+mode)).props('flat no-caps')
                    ui.button('Load',   on_click=lambda mode=mode: ui.navigate.to('/'+mode)).props('flat no-caps')
                    ui.button('Resume', on_click=lambda mode=mode: ui.navigate.to('/'+mode)).props('flat no-caps')
    
    ui.label('Documentation').classes('text-h6')
    ui.separator()
    with ui.row(align_items='center'):
        with ui.card().tight().classes('w-80'):
            with ui.image(hrap_root/'gui'/'thumbnails'/'validation.jpg'):
                with ui.element('div').classes('absolute-bottom text-white p-2'):
                    ui.label('Validation Cases').classes('text-h6')
                    ui.label('Proof of utility double as usage examples').classes('text-subtitle2')
            with ui.card_section():
                ui.button('View', on_click=lambda mode=mode: ui.navigate.to('/validation')).props('flat no-caps')
        with ui.card().tight().classes('w-80'):
            with ui.image(hrap_root/'gui'/'thumbnails'/'api_docs.jpg'):
                with ui.element('div').classes('absolute-bottom text-white p-2'):
                    ui.label('API Documentation').classes('text-h6')
                    ui.label('Our Python interface is flexible and extensible').classes('text-subtitle2')
            with ui.card_section():
                ui.button('View', on_click=lambda mode=mode: ui.navigate.to('/api_docs')).props('flat no-caps')
                ui.button('Open in Browser', on_click=lambda mode=mode: ui.navigate.to('https://github.com/rnickel1/HRAP_Source', new_tab=True)).props('flat no-caps')
    
    ui.label('Settings').classes('text-h6')
    ui.separator()
    
    # ui.label('Your files are preriodically saved')
    
    # TODO: autosave, 
    dark = ui.dark_mode()
    dark.enable()
    def on_theme_change(e):
        match e.value:
            case 'Dark':
                dark.enable()
            case 'Light':
                dark.disable()
    theme = ui.toggle(['Dark', 'Light', 'Genesis', 'Yellow Babber'], value='Dark', on_change=on_theme_change)

def hybrid():
    with ui.row(align_items='center'):
        ox_slider = TankSlider('Nitrous Oxide')
        # tank_slider.bind_value(tank_value, 'value')
        # tank_value = ui.number('Tank Value', value=0.5).props('color=blue')#.bind_value(tank_slider, 'value')
        ui.button('Jiggle', on_click=lambda: ox_slider.run_method('jiggle')).props('outline')

def liquid():
    with ui.row(align_items='center'):
        ox_slider = TankSlider('Nitrous Oxide')
        fu_slider = TankSlider('Ethanol')

def chem():
    with ui.row(align_items='center'):
        ni = ui.number(label='Number', value=3.1415927, format='%.2f')#, suffix='m')
          # on_change=lambda e: result.set_text(f'you entered: {e.value}'))
        # 1. Track the state of the selection
        selection = {'unit': 'USD'}
        with ni.add_slot('append'):
            unit_button = ui.button().props('flat color=grey-4 dense').bind_text_from(selection, 'unit')
        
            # 3. Add the dropdown menu
            with ui.menu() as menu:
                def select_unit(new_unit):
                    selection['unit'] = new_unit
                    # menu.close()
                
                ui.menu_item('USD', on_click=lambda: select_unit('USD'))
                ui.menu_item('EUR', on_click=lambda: select_unit('EUR'))
                ui.menu_item('GBP<sup>3</sup>', on_click=lambda: select_unit('GBP'))
                with ui.menu_item(on_click=lambda: select_unit('Test')):
                    ui.html('m<sup>3</sup>')
                    # ui.markdown(r'm<sup>3</sup>')
                    # ui.markdown(r'$m^3$', extras=['latex']) # needs latex2mathml
        #     with ui.button(icon='arrow_drop_down', color='grey-7').props('flat round'):
        #         with ui.menu() as menu:
        #             ui.menu_item('USD', on_click=lambda: ui.notify('Switched to USD'))
        #             ui.menu_item('EUR', on_click=lambda: ui.notify('Switched to EUR'))
        #             ui.menu_item('GBP', on_click=lambda: ui.notify('Switched to GBP'))
        ui.number(label='O/F', value=4.0, min=0.01, max=100.0, step=0.1)

# Open file dialog then save
async def begin_save():
    result = await app.native.main_window.create_file_dialog(webview.FileDialog.SAVE, allow_multiple=False, directory='/', save_filename='motor.hrap')
    if result != None:
        print('saving', result)
        file_path = result[0]
    # print('saving', active_file)
    # if active_file == None: # Save as
        # dpg.show_item('save_as')
    # else:
        # save_config(active_file)

async def begin_load():
    result = await app.native.main_window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False, directory='/')
    if result != None:
        print('loading', result)
        file_path = result[0]

def root():
    # https://nicegui.io/documentation/sub_pages
    pages = ui.sub_pages()
    # footer = ui.label()
    pages.add('/', lambda: landing())
    pages.add('/hybrid', lambda: hybrid())
    pages.add('/liquid', lambda: liquid())
    pages.add('/chem', lambda: chem())
    # pages.add('/other', lambda: other(footer))
    
    
    # https://nicegui.io/documentation/page_layout
    with ui.header(elevated=True).style('background-color: #3874c8').classes('h-10 gap-0 items-center justify-between'):
        with ui.row(align_items='center').classes('h-full gap-0 ml-[-14px] mt-[-28px]'): # Negative margins to cancel global page margin, disable gap between children
            ui.button('Home', on_click=lambda: ui.navigate.to('/')).props('flat no-caps icon="navigation" color="white"')
            with ui.button('File').props('flat no-caps color="white"'):
                with ui.menu() as menu:
                    ui.menu_item('Load', begin_load)
                    ui.menu_item('Save', begin_save)
                    # # ui.button('Save As')
        
        # with ui.row(align_items='center'):
            # ui.button('Hybrid Engine', on_click=lambda: ui.navigate.to('/hybrid'))
            # ui.button('Liquid Engine', on_click=lambda: ui.navigate.to('/liquid'))
            # ui.button('Solid Motor', on_click=lambda: ui.navigate.to('/solid'))
            # ui.button('Chemistry', on_click=lambda: ui.navigate.to('/chem'))
        # ui.button(on_click=lambda: right_drawer.toggle(), icon='menu').props('flat color=white')

# TODO: any way to autosave on reload?
# Direct call is if ran via "python .\hrap\gui\main.py" instead of "hrap," which enables development features such as auto reload on file change
def main(is_direct_call=False):
    global window, webview #, __name__
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='Open debug, i.e. inspect-element, window', action='store_true')
    parser.add_argument('--host', help='Host rather than displaying in native window', action='store_true')
    # parser.add_argument('--reload', help='Reload when script are changes detecte', action='store_true') # Not possible unless direct...
    # parser.add_argument('--path',  help='Data and plots are written within ./results/nnnfgp/[path]', type=str, default='')
    args = parser.parse_args()
        
    # landing_html = load_hrap_page
    # window = webview.create_window('HRAP', str(hrap_root/'gui'/'frontend'/'dist'/'index.html'))
    
    # https://nicegui.io/documentation/page_layout
    
    # ui.run(root=root, title='HRAP', favicon='🚀', reload=is_direct_call, uvicorn_reload_includes='*.py,*.js,*.vue')
    if args.host:
        ui.run(root=root, title='HRAP', favicon='🚀', reload=is_direct_call, uvicorn_reload_includes='*.py,*.js,*.vue')
    else:
        # both raw webview and nicegui take around 5s to start up...
        import webview # TODO: in deps?
        # from qtpy.QtWebChannel import QWebChannel # TODO: this should be found
        ui.run(native=True, port=native.find_open_port(), root=root, title='HRAP', favicon='🚀', reload=is_direct_call, uvicorn_reload_includes='*.py,*.js,*.vue')
    # ui.run(native=True, reload=is_direct_call, port=native.find_open_port())

if __name__ in {'__main__', '__mp_main__'}: main(is_direct_call=True)
# if __name__ == '__main__': main(is_direct_call=True)
