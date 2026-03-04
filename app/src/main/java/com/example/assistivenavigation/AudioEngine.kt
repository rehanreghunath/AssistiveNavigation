package com.example.assistivenavigation

class AudioEngine {

    companion object {
        init {
            System.loadLibrary("audio_engine")
        }
    }

    external fun nativeStart()
    external fun nativeStop()
    external fun nativeSetSpatialParams(azimuth: Float, distance: Float)
}