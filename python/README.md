# Files
- `mrep.py`: 2D (curve) exploration
- `mrep3.py`: 3D (surface) raytracing (this is probably what you're looking for)
- `teapot.bpt`: B-rep patches for the Utah Teapot
- `raybox.c`: C function to do fast ray-box hit testing

# Dependencies and compilation
- Install Python, NumPy, SciPy, Matplotlib
- Install [`guptri_py`](https://github.com/mwageringel/guptri_py)
  (requires a Fortran compiler to build)
- Run `make` to build the ray-box test function (written in C for speed)
