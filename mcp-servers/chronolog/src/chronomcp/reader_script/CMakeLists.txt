cmake_minimum_required(VERSION 3.25)
project(HDF5SingleFileReader LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CHRONOLOG_ROOT "$ENV{HOME}/ChronoLog" CACHE PATH "ChronoLog source checkout")
set(CHRONOLOG_BUILD "$ENV{HOME}/chronolog-build/Debug" CACHE PATH "ChronoLog build directory")
find_package(Thallium REQUIRED)
find_package(spdlog REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSON_C REQUIRED IMPORTED_TARGET json-c)
find_library(CHRONOLOG_CLIENT_LIBRARY NAMES chronolog_client
    HINTS "${CHRONOLOG_BUILD}/Client/cpp" "${CHRONOLOG_BUILD}/lib" REQUIRED)

add_executable(hdf5_file_reader
    reader.cpp
    ${CHRONOLOG_ROOT}/ChronoPlayer/src/HDF5ArchiveReadingAgent.cpp
    ${CHRONOLOG_ROOT}/chrono_common/src/ConfigurationManager.cpp
    ${CHRONOLOG_ROOT}/chrono_common/src/StoryChunk.cpp
    ${CHRONOLOG_ROOT}/chrono_common/src/StoryChunkWriter.cpp)
target_include_directories(hdf5_file_reader PRIVATE
    ${CHRONOLOG_ROOT}/ChronoPlayer/include
    ${CHRONOLOG_ROOT}/chrono_common/include
    ${CHRONOLOG_ROOT}/Client/cpp/include
    ${HDF5_INCLUDE_DIRS})
target_link_libraries(hdf5_file_reader PRIVATE
    ${CHRONOLOG_CLIENT_LIBRARY} thallium spdlog::spdlog ${HDF5_LIBRARIES} PkgConfig::JSON_C)
set_target_properties(hdf5_file_reader PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS hdf5_file_reader RUNTIME DESTINATION bin)
