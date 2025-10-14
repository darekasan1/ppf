cmake_minimum_required(VERSION 3.15)
project(ppf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  midifile
  GIT_REPOSITORY https://github.com/craigsapp/midifile.git
  GIT_TAG        master
)

# ★★★【修正点】Midifileのツールをビルドしないように設定 ★★★
FetchContent_GetProperties(midifile)
if(NOT midifile_POPULATED)
  FetchContent_Populate(midifile)
  add_subdirectory(${midifile_SOURCE_DIR} ${midifile_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

add_executable(ppf main.cpp synth_engine.cpp)

target_include_directories(ppf PRIVATE ${midifile_SOURCE_DIR}/include)

target_link_libraries(ppf PRIVATE midifile)

if(CMAKE_COMPILER_IS_GNU_CXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(ppf PRIVATE -O3 -pthread)
    target_link_options(ppf PRIVATE -pthread)
endif()
