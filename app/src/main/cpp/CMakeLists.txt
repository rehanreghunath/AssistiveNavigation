cmake_minimum_required(VERSION 3.22.1)

project(audio_engine)

add_library(audio_engine SHARED
        AudioEngine.cpp)

find_package(oboe REQUIRED CONFIG)

find_library(log-lib log)

target_link_libraries(audio_engine
        oboe::oboe
        ${log-lib})