#pragma once
#include <oboe/Oboe.h>
#include <atomic>
#include <cstdint>
#include <cstring>

class AudioEngine : public oboe::AudioStreamCallback {
public:
    AudioEngine();
    ~AudioEngine();

    bool start();
    void stop();

    // azimuthDeg : -90 (full left) … 0 (centre) … +90 (full right)
    // distanceM  : metres to the source
    void setSpatialParams(float azimuthDeg, float distanceM);

    oboe::DataCallbackResult onAudioReady(
            oboe::AudioStream* stream,
            void*              audioData,
            int32_t            numFrames) override;

private:
    oboe::AudioStream* stream = nullptr;

    // Written from the Kotlin thread, read from the audio thread
    std::atomic<float> targetAzimuth {0.0f};
    std::atomic<float> targetDistance{2.0f};

    // Smoothed copies used only on the audio thread (no locking needed)
    float currentAzimuth  = 0.0f;
    float currentDistance = 2.0f;

    // ---- White noise ----
    // xorshift32 is a fast, good-quality pseudo-random number generator.
    // It is far cheaper than std::mt19937 in a tight audio loop.
    uint32_t rngState = 2463534242u;  // any non-zero seed

    inline float nextNoise() noexcept {
        rngState ^= rngState << 13u;
        rngState ^= rngState >> 17u;
        rngState ^= rngState << 5u;
        // Map uint32 → [-1, +1]
        return static_cast<float>(rngState) * (1.0f / 2147483648.0f) - 1.0f;
    }

    // ---- ITD delay line ----
    // Max ITD at 90° ≈ 31 samples at 48 kHz. 64 gives comfortable headroom.
    static constexpr int MAX_DELAY = 64;
    float delayLine[MAX_DELAY];   // zero-filled in constructor
    int   delayWriteIdx = 0;

    // ---- Head-shadow IIR filter state ----
    // One state variable per ear. Only the far ear gets filtered.
    float filterStateL = 0.0f;
    float filterStateR = 0.0f;

    // ---- Fade-in envelope ----
    // Ramps from 0 → 1 over ~40 ms so there's no click when the stream starts.
    float envelope = 0.0f;
    static constexpr float ENVELOPE_ATTACK = 0.0005f; // per sample

    // ---- Binaural parameter bundle (computed once per buffer) ----
    struct BinauralParams {
        float gainL;            // left channel amplitude  [0..1]
        float gainR;            // right channel amplitude [0..1]
        int   itdSamples;       // how many samples to delay the far ear
        float headShadowAlpha;  // IIR coefficient for the far-ear filter
        // 1.0 = no filter, ~0.12 = heavy low-pass
        bool  leftIsFar;        // true when source is on the right side
    };

    BinauralParams computeBinaural(float azDeg, float distM) noexcept;
};