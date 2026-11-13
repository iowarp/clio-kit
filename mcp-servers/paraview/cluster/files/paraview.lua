-- ParaView, headless (OSMesa), from conda-forge. Loaded with
-- `module load paraview`; the version is this file's name.
local version = myModuleVersion()
local root = pathJoin("/opt/apps/paraview", version)
help("ParaView " .. version .. ", headless (OSMesa software rendering), Python 3.12")
whatis("Name: ParaView")
whatis("Version: " .. version)
prepend_path("PATH", pathJoin(root, "bin"))
setenv("PARAVIEW_HOME", root)
setenv("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
