# 项目总配置
cmake_minimum_required(VERSION 3.12)
project(Vision-2024)
# 设置C++标准版本
set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_NAME vision-2024)

# 设置编译选项
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
  set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
  add_compile_options(-stdlib=libc++)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_COMPILER g++) # Ubuntu 20.04 or 22.04
  elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_COMPILER g++-8) # Ubuntu 18.04
  endif()
endif()



# 设置优化参数
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  add_compile_definitions(DEBUG)
  add_compile_options(-O0)
elseif(CMAKE_BUILD_TYPE STREQUAL Release)
  add_compile_definitions(RELEASE)
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    add_compile_options(-O3 -march=x86-64)
  elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_options(-O3 -march=native)
  endif()
endif()

# 设置输出位置
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 添加可执行文件
add_executable(${EXECUTABLE_NAME} main.cpp)

# 编译模块 注意顺序问题
add_subdirectory(modules/common)
add_subdirectory(modules/autoaim)
add_subdirectory(modules/core)

# 链接库主程序
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  target_link_libraries(
    ${EXECUTABLE_NAME} 
    PUBLIC srm_core
  )
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
  target_link_libraries(
    ${EXECUTABLE_NAME} 
    PUBLIC -Wl,-no-as-needed srm_core
  )
endif()