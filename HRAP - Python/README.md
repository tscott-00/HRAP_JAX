## User Installation Instructions (Windows/Linux/Mac)
First install Python https://www.python.org/downloads/. Be sure to select add to PATH. Version >=3.10 is required.

Then open your terminal and execute:

```
python -m pip install --upgrade pip

python -m pip install hrap
```

If python was not found, try typing python3 instead.

To update hrap when new features become available, simply run

```
python -m pip install --upgrade hrap
```

To uninstall hrap,

```
python -m pip uninstall hrap
```

## Running HRAP
To start the GUI, simply run the command from any directory:
```
hrap
```
If you wish to run HRAP non-natively, host locally then open in any web browser
```
hrap --host
```

## Alternative installations
If you wish to run the sim on GPU, install JAX for GPU
```
python -m pip install jax[gpu]
```
If you are on Linux and run into errors due to lack of a backend, install PyQT6 dependencies
```
python -m pip install pywebview[qt]
```
You may have to uninstall PyQT5 to avoid conflicts. If you wish to use PyQT5 or need an alternative backend then see https://pywebview.flowrl.com/guide/installation.html or view HRAP in a web browser (recommended if issues persist).

## Advanced Usage
If you are interested in advanced usage such as design optimization, uncertainty quantification, adding custom engine models, etc. then the Python API is for you.
See the "HRAP - Python/hrap/examples" directories for several basic API examples.

HRAP Python utilizes JAX for Just-In-Time (JIT) compilation, enabling fast execution speeds while maximizing compatibility.
The helper functions in HRAP minimize the technical knowledge of JAX and Python required to make modifications.
Still, the functional programming style may be unfamiliar to some, particularly with loops and conditionals. See https://docs.jax.dev/en/latest/control-flow.html#control-flow.

## Developer Installation Instructions (Windows/Linux/Mac)
This is only necessary if you wish to contribute to the official HRAP repository.
You can script your own custom functionality with the "User Installation" via the API.
This is an alternative to "User Installation." Please uninstall hrap or use a new venv before switching to the "Developer Installation."

After cloning the repo,

```
cd "HRAP_JAX/HRAP - Python/"

python -m pip install -ve ./
```

You can now run the GUI as usual. Your local modifications to the hrap code will be reflected.
