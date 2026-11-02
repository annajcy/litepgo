# The extension is built against the project's virtual environment: the active
# one ($VIRTUAL_ENV, e.g. under uv run), else .venv in the source tree. Pass
# -DPython_EXECUTABLE=... to choose another.
if(NOT DEFINED Python_EXECUTABLE)
    if(DEFINED ENV{VIRTUAL_ENV})
        set(_litepgo_venv "$ENV{VIRTUAL_ENV}")
    else()
        set(_litepgo_venv "${PROJECT_SOURCE_DIR}/.venv")
    endif()
    if(WIN32)
        set(_litepgo_python "${_litepgo_venv}/Scripts/python.exe")
    else()
        set(_litepgo_python "${_litepgo_venv}/bin/python")
    endif()
    if(NOT EXISTS "${_litepgo_python}")
        message(FATAL_ERROR "No virtual environment at ${_litepgo_venv}. Run `uv sync --locked` first, "
                            "or set Python_EXECUTABLE.")
    endif()
    set(Python_EXECUTABLE "${_litepgo_python}" CACHE FILEPATH "Python interpreter of the virtual environment")
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
