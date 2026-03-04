#pragma once

#include <oboe/Oboe.h>
#include <atomic>

class AudioEngine : public oboe::AudioStreamCallback {

public:
    AudioEngine();
    ~AudioEngine();

    bool start();
    void stop();

    void setSpatialParams(float azimuthDeg, float distanceMeters);

    // Oboe callback
    oboe::DataCallbackResult
    onAudioReady(oboe::AudioStream* stream,
                 void* audioData,
                 int32_t numFrames) override;

private:
    oboe::AudioStream* stream = nullptr;

    std::atomic<float> azimuth {0.0f};
    std::atomic<float> distance {2.0f};

    float phase = 0.0f;

    void computeStereoGains(float& left, float& right);
};