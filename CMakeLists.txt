cmake_minimum_required(VERSION 3.20)
project(PatMax2D VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt Creator 未设置 OpenCV_DIR 时，使用本机与运行脚本一致的 OpenCV 4.8.0。
# 显式指定的 OpenCV_DIR 仍优先，便于在其他机器或版本上构建。
if(NOT OpenCV_DIR AND EXISTS "D:/opencv4.8.0/opencv/build/x64/vc16/lib/OpenCVConfig.cmake")
    set(OpenCV_DIR "D:/opencv4.8.0/opencv/build/x64/vc16/lib" CACHE PATH "OpenCV CMake package directory" FORCE)
endif()
find_package(OpenCV 4.8.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Qt5 5.15 REQUIRED COMPONENTS Widgets Gui)

add_library(patmax_core src/ShapeMatcher.cpp)
target_include_directories(patmax_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(patmax_core PUBLIC ${OpenCV_LIBS})
target_compile_features(patmax_core PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(patmax_core PRIVATE /utf-8)
endif()

# 从 Designer 的 .ui 文件生成头文件；生成文件属于构建产物，不要手工修改。
set(PATMAX_UI_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/ui_MainWindow.h")
add_custom_command(
    OUTPUT "${PATMAX_UI_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "$<TARGET_FILE:Qt5::uic>" "${CMAKE_CURRENT_SOURCE_DIR}/src/MainWindow.ui" -o "${PATMAX_UI_HEADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/MainWindow.ui"
    VERBATIM)
add_executable(patmax_gui WIN32 src/main.cpp src/MainWindow.cpp src/ImageView.cpp "${PATMAX_UI_HEADER}")
target_include_directories(patmax_gui PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(patmax_gui PRIVATE patmax_core Qt5::Widgets Qt5::Gui)
if(MSVC)
    target_compile_options(patmax_gui PRIVATE /utf-8)
endif()

add_executable(patmax_smoke tests/smoke.cpp)
target_link_libraries(patmax_smoke PRIVATE patmax_core ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(patmax_smoke PRIVATE /utf-8)
endif()

# 手工基准程序：用实际大图检查层数、阶段耗时和最终定位。
add_executable(patmax_bench tests/large_case_bench.cpp)
target_link_libraries(patmax_bench PRIVATE patmax_core Qt5::Gui)
if(MSVC)
    target_compile_options(patmax_bench PRIVATE /utf-8)
endif()

enable_testing()
set(PATMAX_RINGS_DIR "C:/Program Files/MVTec/HALCON-23.05-Progress/examples/images/rings"
    CACHE PATH "Optional directory containing HALCON rings_01.png and rings_06.png")
if(EXISTS "${PATMAX_RINGS_DIR}/rings_01.png")
    add_test(NAME rings_smoke COMMAND patmax_smoke "${PATMAX_RINGS_DIR}")
endif()
