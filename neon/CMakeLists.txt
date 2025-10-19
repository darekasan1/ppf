cmake_minimum_required(VERSION 3.15)
project(ppf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ポリシー設定（警告を消すため）
cmake_policy(SET CMP0069 NEW)

# MIDIファイルライブラリの取得
include(FetchContent)
FetchContent_Declare(
  midifile
  GIT_REPOSITORY https://github.com/craigsapp/midifile.git
  GIT_TAG        master
)

# Midifileのツールをビルドしないように設定
set(BUILD_PROGRAMS OFF CACHE BOOL "Build midifile programs" FORCE)

FetchContent_GetProperties(midifile)
if(NOT midifile_POPULATED)
  FetchContent_Populate(midifile)
  add_subdirectory(${midifile_SOURCE_DIR} ${midifile_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# 実行ファイルの作成
add_executable(ppf main.cpp synth_engine.cpp)

# インクルードディレクトリ
target_include_directories(ppf PRIVATE ${midifile_SOURCE_DIR}/include)

# ライブラリのリンク
target_link_libraries(ppf PRIVATE midifile)

# コンパイラ最適化オプション
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # 基本的な最適化
    target_compile_options(ppf PRIVATE 
        -O3                      # 最高レベルの最適化
        -pthread                 # マルチスレッド対応
    )
    
    target_link_options(ppf PRIVATE -pthread)
    
    # ARM/NEON用の最適化
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        message(STATUS "ARM64アーキテクチャを検出しました - NEON最適化を有効化")
        
        target_compile_options(ppf PRIVATE
            -march=native        # CPUの全機能を使用
            -mtune=native        # CPUに合わせたチューニング
            -funroll-loops       # ループ展開
            -fomit-frame-pointer # フレームポインタ省略
        )
        
    # x86/x64用の最適化（参考）
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
        message(STATUS "x86/x64アーキテクチャを検出しました - SSE/AVX最適化を有効化")
        
        target_compile_options(ppf PRIVATE
            -march=native
            -mtune=native
            -msse4.2
            -mavx2
            -mfma
            -ffast-math
            -funroll-loops
            -fomit-frame-pointer
            -fno-math-errno
        )
    endif()
    
    # Release/RelWithDebInfo ビルドでの最適化（LTOを削除）
    # Termux/Androidでは-fltoが問題を起こすため無効化
    
endif()

# MSVCの場合
if(MSVC)
    target_compile_options(ppf PRIVATE
        /O2                      # 最適化
        /GL                      # プログラム全体の最適化
        /arch:AVX2              # AVX2命令
        /fp:fast                # 高速浮動小数点
    )
    
    target_link_options(ppf PRIVATE
        /LTCG                   # リンク時コード生成
    )
endif()

# ビルドタイプが指定されていない場合はReleaseに設定
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ビルド情報の表示
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "LTO enabled: ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
