import sys
sys.path.insert(1, '../HRAP/')
import time

import scipy
import numpy as np
from pathlib import Path

# import matplotlib.pyplot as plt

import dearpygui.dearpygui as dpg

from jax.scipy.interpolate import RegularGridInterpolator

import hrap.core as core
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.sat_nos import *
from hrap.units   import _in, _ft

def main():
    jax.config.update("jax_enable_x64", True)
    
    # See https://github.com/hoffstadt/DearPyGui

    def save_callback():
        print('Save Clicked')

    dpg.create_context()
    dpg.create_viewport(title='HRAP', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.set_viewport_vsync(False)
    # dpg.set_viewport_vsync(True)

    def resize_windows():
        # Get the size of the main window
        main_width, main_height = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()

        # Update the size and position of each window based on the main window's size
        dpg.set_item_width ('Tank', main_width // 2)
        dpg.set_item_height('Tank', main_height // 3)
        dpg.set_item_pos   ('Tank', [0, 0])

        dpg.set_item_width ('Grain', main_width // 2)
        dpg.set_item_height('Grain', main_height // 3)
        dpg.set_item_pos   ('Grain', [0, main_height // 3])
        
        dpg.set_item_width ('Chamber', main_width // 2)
        dpg.set_item_height('Chamber', main_height // 3)
        dpg.set_item_pos   ('Chamber', [0, 2 * main_height // 3])
        
        dpg.set_item_width ('General', main_width // 2)
        dpg.set_item_height('General', main_height // 3)
        dpg.set_item_pos   ('General', [main_width // 2, 0])

        dpg.set_item_width ('Preview', main_width // 2)
        dpg.set_item_height('Preview', main_height // 3)
        dpg.set_item_pos   ('Preview', [main_width // 2, main_height // 3])
        
        dpg.set_item_width ('Nozzle', main_width // 2)
        dpg.set_item_height('Nozzle', main_height // 3)
        dpg.set_item_pos   ('Nozzle', [main_width // 2, 2 * main_height // 3])

    # First row
    settings = { 'no_move': True, 'no_collapse': True, 'no_resize': True, 'no_close': True }

    with dpg.window(tag='General', label='General', **settings):
        dpg.add_input_text(label='file name')
        dpg.add_button(label='Save', callback=save_callback)

    with dpg.window(tag='Preview', label='Preview', **settings):
        # dpg.add_text('Bottom Right Section')
        # dpg.add_simple_plot(label="Simple Plot", min_scale=-1.0, max_scale=1.0, height=300, tag="plot")
        # create plot
        with dpg.plot(label="Line Series", height=300, width=800):
            # optionally create legend
            dpg.add_plot_legend()

            # REQUIRED: create x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label="t (s)")
            dpg.add_plot_axis(dpg.mvYAxis, label="Thrust (N)", tag="y_axis")

            # series belong to a y axis
            dpg.add_line_series([], [], label="Trust", parent="y_axis", tag="series_tag")


    # with dpg.window(tag='Tank', label='Tank', **settings):
        # dpg.add_text('Top Right Section')

    # with dpg.window(tag='Grain', label='Grain', **settings):
        # dpg.add_text('Top Right Section')

    tnk_config = {
        'Volume': {
            'type': float,
            'key': 'V',
            'min': 1E-9,
            'max': 1.0,
            'default': (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
            'step': 1E-4,
            'decimal': 6,
        },
        'Injector CdA': {
            'type': float,
            'key': 'inj_CdA',
            'min': 1E-9,
            'default': 0.5 * (np.pi/4 * 0.5**2 * _in**2),
            'step': 1E-6,
            'decimal': 6,
        },
        'Oxidizer Mass': {
            'type': float,
            'key': 'm_ox',
            'min': 1E-3,
            'max': 1E+3,
            'default': 14.0,
            'step': 1E-1,
            'decimal': 3,
        },
        # V = (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
        # inj_CdA= 0.5 * (np.pi/4 * 0.5**2 * _in**2),
        # m_ox=14.0
    }
    
    # TODO: add info descriptions!
    cmbr_config = {
        'Base Volume [m^3]': {
            'type': float,
            'key': 'V0',
            'min': 0.0,
            'step': 1E-4,
            'decimal': 6,
        },
    }
    
    grain_config = {
        'Outer diamater': {
            'type': float,
            'key': 'OD',
            'min': 0.001,
            'default': 5.0 * _in,
            'step': 1E-3,
            'decimal': 3,
        },
        'Length': {
            'type': float,
            'key': 'OD',
            'min': 0.001,
            'default': 5.0 * _ft,
            'step': 1E-2,
            'decimal': 3,
        },
        'Fixed O/F ratio': {
            'type': float,
            'key': 'OF',
            'min': 0.01,
            'max': 100.0,
            'default': 5.0,
            'step': 1E-1,
            'decimal': 2,
        },
    }

    noz_config = {
        'Discharge Coefficient': {
            'type': float,
            'key': 'Cd',
            'min': 0.01,
            'max': 1.0,
            'default': 0.9,
            'step': 1E-2,
            'decimal': 2,
        },
        'Efficiency': {
            'type': float,
            'key': 'eff',
            'min': 0.01,
            'max': 1.0,
            'default': 0.9,
            'step': 1E-2,
            'decimal': 2,
        },
        'Throat Diameter [m]': {
            'type': float,
            'key': 'thrt',
            'min': 0.001,
            'default': 1.5 * _in,
            'step': 1E-3,
            'decimal': 3,
        },
        'Exit/Throat Area Ratio': {
            'type': float,
            'key': 'ER',
            'min': 1.001,
            'default': 5.0,
            'step': 1E-1,
            'decimal': 3,
        },
    }

    def make_part_window(name, part_config):
        for key in part_config:
            part_config[key]['uuid'] = dpg.generate_uuid()
        # print(name)
        with dpg.window(tag=name, label=name, **settings):
            for title, props in part_config.items():
                if props['type'] == float:
                    decimal = props['decimal'] if 'decimal' in props else 3
                    dpg.add_input_float(label=title, step=props['step'], format=f'%.{decimal}f', tag=props['uuid'])
                    if 'default' in props:
                        dpg.set_value(props['uuid'], props['default'])
                    # dpg.add_text(key)

    make_part_window('Tank', tnk_config)
    make_part_window('Grain', grain_config)
    make_part_window('Chamber', cmbr_config)
    make_part_window('Nozzle', noz_config)
    part_configs = { 'cmbr': cmbr_config, 'noz': noz_config, 'tnk': tnk_config, 'grn': grain_config }


    # with dpg.window(tag='Nozzle', label='Nozzle', **settings):
        # dpg.add_text('Bottom Right Section')

    # chem = scipy.io.loadmat('../../propellant_configs/HTPB.mat')
    # import pkgutils
    # data_dir = Path(pkgutils.resolve_name('hrap.tank').__file__).parent
    # data_path = Path(data_dir , 'HTPB.mat')
    from importlib.resources import files as imp_files
    chem = scipy.io.loadmat(str(imp_files('hrap').joinpath('HTPB.mat')))
    
    chem = chem['s'][0][0]
    chem_OF = chem[1].ravel()
    chem_Pc = chem[0].ravel()
    chem_k = chem[2]
    chem_M = chem[3]
    chem_T = chem[4]

    chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k, fill_value=1.4)
    chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M, fill_value=29.0)
    chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T, fill_value=293.0)

    # Initialization
    tnk = make_sat_tank(
        get_sat_nos_props,
        V = (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
        inj_CdA= 0.5 * (np.pi/4 * 0.5**2 * _in**2),
        m_ox=1,#14.0, # TODO: init limit
        # m_ox = 3.0,
    )
    # print('INJ TEST', 0.5 * (np.pi/4 * 0.5**2 * _in**2))

    shape = make_circle_shape(
        ID = 2.5 * _in,
    )
    grn = make_constOF_grain(
        shape,
        OF = 3.0,
        OD = 5.0 * _in,
        L = 4.0 * _ft,
    )

    cmbr = make_chamber(
    )

    noz = make_cd_nozzle(
        thrt = 1.5 * _in, # Throat diameter
        ER = 5.0,         # Exit/throat area ratio
    )

    s, x, method = core.make_engine(
        tnk, grn, cmbr, noz,
        chem_interp_k=chem_interp_k, chem_interp_M=chem_interp_M, chem_interp_T=chem_interp_T,
        Pa=101e3,
    )

    fire_engine = core.make_integrator(
        # core.step_rk4,
        core.step_fe,
        method,
    )

    resize_windows()
    dpg.set_viewport_resize_callback(resize_windows)
    
    upd_max_fps = 4
    upd_wall_dT = 1 / upd_max_fps # minimum time between relevant engine updates
    upd_wall_t = time.time() - 2*upd_wall_dT # time of last update
    upd_due = True
    
    max_fps = 24
    frame_wall_dT = 1/max_fps
    

    # dpg.add_text('Output')
    # dpg.add_input_text(label='file name')
    # dpg.add_button(label='Save', callback=save_callback)
    # dpg.add_slider_float(label='float')

    dpg.show_viewport()

    _unpack_engine = jax.jit(partial(core.unpack_engine, method=method))

    fps_wall_t = time.time()
    fps_i = 0
    while dpg.is_dearpygui_running():
        wall_t = time.time()
        
        # t1 = time.time()
        # TODO: can be done in callbacks?
        for part_name, part_config in part_configs.items():
            for key, props in part_config.items():
                val = dpg.get_value(props['uuid'])
                if 'min' in props and val < props['min']:
                    dpg.set_value(props['uuid'], props['min'])
                if 'max' in props and val > props['max']:
                    dpg.set_value(props['uuid'], props['max'])
            
            for value_config in part_config.values():
                # print('set', part_name+'_'+value_config['key'], s[part_name+'_'+value_config['key']], '->', dpg.get_value(value_config['uuid']))
                k = part_name+'_'+value_config['key']
                v = dpg.get_value(value_config['uuid'])
                if k in s:
                    if s[k] != v:
                        s[k] = v
                        upd_due = True
                elif k in method['xmap']:
                    if x[method['xmap'][k]] != v:
                        x = x.at[method['xmap'][k]].set(v)
                        upd_due = True
                else:
                    print('ERROR:', k, 'is nowhere!')
        # t2 = time.time()
        # print('v check took', t2-t1)
        
        
        # s['noz_eff'] = dpg.get_value(noz_config['Efficiency']['uuid'])
        # s['noz_thrt'] = dpg.get_value(noz_config['Throat Diameter [m]']['uuid'])
        if upd_due and wall_t - upd_wall_t >= upd_wall_dT:
            upd_due = False
            upd_wall_t = wall_t
        
            T = 10.0
            t10 = time.time()
            t, x1, xstack = fire_engine(s, x, dt=1E-3, T=T)
            jax.block_until_ready(xstack)
            # tnk, grn, cmbr, noz = _unpack_engine(s, xstack)
            
            N_t = xstack.shape[0]
            t2 = time.time()

            thrust = xstack[:,method['xmap']['noz_thrust']]
            # print(t.shape) # What happened to arr?
            # dpg.set_value('series_tag', [np.asarray(t[::10]), np.asarray(thrust[::10])])
            dpg.set_value('series_tag', [np.linspace(0.0, T, N_t//10), np.asarray(thrust[::10])])
            print('max engine fps', 1/(t2-t10))
            # dpg.set_value('series_tag', [np.linspace(0.0, T, N_t), np.asarray(noz['thrust'])])

        dpg.render_dearpygui_frame()
        
        wall_t_end = time.time()
        extra_time = frame_wall_dT - (wall_t_end - wall_t)
        if extra_time > 0.0:
            time.sleep(extra_time)

        # TODO: show on frame somewhere, or use dpg.get_frame_rate()
        # fps_i += 1
        # if wall_t >= fps_wall_t + 1.0:
            # print('FPS:', fps_i)#, '  freq', int(1/(t2-t1)))
            # fps_i = 0
            # fps_wall_t = wall_t # TODO: + modulus

    # dpg.start_dearpygui()
    dpg.destroy_context()
