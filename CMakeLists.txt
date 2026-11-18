#----------------------------------------------------------------------------
# High-granularity toy calorimeter: the Geant4 simulation as a Python module (minicalo).
#
# Builds against a current Geant4 (tested with 11.4.3) and Pythia 8.3 (tested with 8.318). pybind11 comes from
# the Python environment (pip install pybind11), jsoncpp from the submodule. No ROOT and no fastjet are needed.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#         -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
#   cmake --build build -j 16
#----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(hg_toy_calorimeter CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# No link-time optimisation. pybind11 adds -flto to Release builds unless this is set; with it, some builds of the
# unchanged code segfaulted while the module was imported (static initialisation, after numpy was loaded) and others
# did not, depending only on where the tree was built. Builds without LTO, and debug builds, never crashed.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

find_package(Geant4 REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Pythia 8: only the include and lib dirs (pythia8-config --cxxflags carries an old -std= flag)
execute_process(COMMAND pythia8-config --includedir OUTPUT_VARIABLE PY8_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND pythia8-config --libdir OUTPUT_VARIABLE PY8_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT PY8_INC)
    message(FATAL_ERROR "pythia8-config not found on PATH")
endif()

# jsoncpp (submodule), static so the module has no extra runtime dependency
set(JSONCPP_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_PKGCONFIG_SUPPORT OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_CMAKE_PACKAGE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_OBJECT_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(jsoncpp EXCLUDE_FROM_ALL)

set(CoreFiles
        src/Calorimeter.cc
        src/PrimariesGenerator.cc
        src/B4DetectorConstruction.cc
        src/B4aActionInitialization.cc
        src/B4aSteppingAction.cc
        src/B4PartGeneratorBase.cc
        src/B4RunAction.cc
        src/B4aEventAction.cc
        src/SensorContainer.cc)

pybind11_add_module(minicalo src/minicalo_library.cc ${CoreFiles})
target_include_directories(minicalo PRIVATE ${PROJECT_SOURCE_DIR}/include ${PY8_INC})
target_link_directories(minicalo PRIVATE ${PY8_LIBDIR})
target_link_libraries(minicalo PRIVATE ${Geant4_LIBRARIES} jsoncpp_static pythia8)
