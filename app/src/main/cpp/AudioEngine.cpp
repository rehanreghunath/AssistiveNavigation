#include "AudioEngine.h"
#include <cmath>
#include <algorithm>
#include <jni.h>

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ---- Constructor / Destructor ----

AudioEngine::AudioEngine() {
    // Zero the delay line so the far ear hears silence (not garbage) at startup
    std::fill(std::begin(delayLine), std::end(delayLine), 0.0f);
}

AudioEngine::~AudioEngine() { stop(); }

// ---- Stream lifecycle ----

bool AudioEngine::start() {
    oboe::AudioStreamBuilder builder;
    builder
            .setDirection(oboe::Direction::Output)
                    // LowLatency asks Oboe to minimise the audio buffer size.
                    // Oboe will also try to open the stream in Exclusive mode (lowest
                    // latency) and fall back to Shared if the device refuses.
            ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
            ->setSharingMode(oboe::SharingMode::Exclusive)
            ->setFormat(oboe::AudioFormat::Float)
            ->setChannelCount(2)       // stereo
            ->setSampleRate(48000)     // standard for Android audio
            ->setCallback(this);

    if (builder.openStream(&stream) != oboe::Result::OK)  return false;
    if (stream->requestStart()      != oboe::Result::OK)  return false;
    return true;
}

void AudioEngine::stop() {
    if (stream) {
        // stop() blocks until the audio callback has finished its current call,
        // so it is safe to delete state after this returns.
        stream->stop();
        stream->close();
        stream = nullptr;
    }
    // Reset stateful audio processing so the next start() is clean
    envelope      = 0.0f;
    filterStateL  = 0.0f;
    filterStateR  = 0.0f;
    delayWriteIdx = 0;
    std::fill(std::begin(delayLine), std::end(delayLine), 0.0f);
}

// ---- Public setter (called from Kotlin thread) ----

void AudioEngine::setSpatialParams(float azimuthDeg, float distanceM) {
    // relaxed ordering is sufficient here: we only need the audio thread to
    // eventually see the new values, not in any strict order.
    targetAzimuth .store(azimuthDeg,  std::memory_order_relaxed);
    targetDistance.store(distanceM,   std::memory_order_relaxed);
}

// ---- Binaural parameter computation ----

AudioEngine::BinauralParams
AudioEngine::computeBinaural(float azDeg, float distM) noexcept {

    // Clamp to the range our model covers
    azDeg = std::max(-90.0f, std::min(90.0f, azDeg));
    const float azRad = azDeg * (M_PI_F / 180.0f);

    // ---- ILD: equal-power panning ----
    // pan ∈ [-1, +1];  angle maps that into [0, π/2] for cos/sin split.
    const float pan   = azDeg / 90.0f;
    const float angle = (pan + 1.0f) * M_PI_F / 4.0f;
    float gainL = cosf(angle);   // decreases as source moves right
    float gainR = sinf(angle);   // increases as source moves right

    // ---- ITD: Woodworth spherical-head model ----
    // The formula models how far sound has to travel around the head to reach
    // the far ear.  r = 0.0875 m (average adult head radius), c = 343 m/s.
    //
    //   ITD_seconds = (r / c) × (sin(θ) + θ)
    //
    // At θ = 90°: ITD ≈ 0.65 ms ≈ 31 samples at 48 kHz.
    // Positive azimuth → source on right → RIGHT ear leads → delay LEFT.
    const float r = 0.0875f, c = 343.0f, fs = 48000.0f;
    const float itdSec = (r / c) * (sinf(azRad) + azRad);  // positive = right
    int itdSamples = static_cast<int>(fabsf(itdSec) * fs + 0.5f);
    itdSamples = std::min(itdSamples, MAX_DELAY - 1);

    // ---- Head-shadow filter coefficient ----
    // At azimuth 0° both ears hear the same thing → no filtering (alpha = 1).
    // At azimuth ±90° the far ear is heavily shadowed → strong low-pass.
    //
    // We linearly interpolate alpha from 1.0 (no shadow) to 0.12 (~900 Hz
    // cutoff at 48 kHz) as |azimuth| goes from 0° to 90°.
    //
    // Why 0.12?  For a first-order IIR  y[n] = α·x[n] + (1-α)·y[n-1],
    //   cutoff ≈ -fs · ln(1 - α) / (2π)
    //   α = 0.12 → cutoff ≈ 920 Hz  (matches psychoacoustic data roughly)
    const float azAbs         = fabsf(azDeg) / 90.0f;  // [0, 1]
    const float headShadowAlpha = 1.0f - azAbs * 0.88f; // 1.0 → 0.12

    // ---- Distance attenuation ----
    // Simple inverse-distance law; clamped so nearby objects aren't deafening.
    const float distGain = 1.0f / std::max(distM, 0.5f);
    gainL *= distGain;
    gainR *= distGain;

    const bool leftIsFar = (azDeg >= 0.0f); // source on right → left is the far ear

    return {gainL, gainR, itdSamples, headShadowAlpha, leftIsFar};
}

// ---- Audio callback (runs on a real-time thread — no allocations, no locks) ----

oboe::DataCallbackResult
AudioEngine::onAudioReady(oboe::AudioStream* /*unused*/,
                          void*    audioData,
                          int32_t  numFrames) {

    float* out = static_cast<float*>(audioData);

    // ---- Smooth parameters toward the latest target (once per buffer) ----
    // This avoids sudden jumps in gain that would cause audible clicks.
    // The factor 0.1 means each buffer moves 10% of the remaining distance.
    const float tAz   = targetAzimuth .load(std::memory_order_relaxed);
    const float tDist = targetDistance.load(std::memory_order_relaxed);
    currentAzimuth  += (tAz   - currentAzimuth)  * 0.1f;
    currentDistance += (tDist - currentDistance) * 0.1f;

    const BinauralParams bp = computeBinaural(currentAzimuth, currentDistance);

    for (int i = 0; i < numFrames; ++i) {

        // ---- Generate white noise sample ----
        // White noise contains all frequencies equally — far more natural for
        // a proximity cue than a sine tone, and much cheaper than speech/music.
        // We scale to 0.35 so the peak level stays well below 0 dBFS.
        const float noise = nextNoise() * 0.35f;

        // ---- ITD delay line ----
        // Write the current sample, then immediately read the far-ear position
        // (which is itdSamples steps behind the write pointer).
        delayLine[delayWriteIdx] = noise;
        const int readIdx = (delayWriteIdx - bp.itdSamples + MAX_DELAY) % MAX_DELAY;
        const float farSample  = delayLine[readIdx];
        const float nearSample = noise;
        delayWriteIdx = (delayWriteIdx + 1) % MAX_DELAY;

        // ---- Head-shadow IIR filter on the far ear ----
        // y[n] = α·x[n] + (1-α)·y[n-1]
        // α close to 1 → minimal filtering (near-frontal sounds)
        // α close to 0 → heavy low-pass (source far to the side)
        float filteredFar;
        if (bp.leftIsFar) {
            filterStateL = bp.headShadowAlpha * farSample
                           + (1.0f - bp.headShadowAlpha) * filterStateL;
            filteredFar  = filterStateL;
        } else {
            filterStateR = bp.headShadowAlpha * farSample
                           + (1.0f - bp.headShadowAlpha) * filterStateR;
            filteredFar  = filterStateR;
        }

        // ---- Mix to stereo output ----
        float sL, sR;
        if (bp.leftIsFar) {
            sL = filteredFar  * bp.gainL;   // left  = far (delayed + filtered)
            sR = nearSample   * bp.gainR;   // right = near
        } else {
            sL = nearSample   * bp.gainL;   // left  = near
            sR = filteredFar  * bp.gainR;   // right = far (delayed + filtered)
        }

        // ---- Fade-in envelope ----
        // Prevents the hard click that occurs when a stream starts at full volume.
        envelope = std::min(1.0f, envelope + ENVELOPE_ATTACK);

        out[i * 2]     = sL * envelope;
        out[i * 2 + 1] = sR * envelope;
    }

    return oboe::DataCallbackResult::Continue;
}

// ---- JNI bindings ----

static AudioEngine* engine = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_com_example_assistivenavigation_AudioEngine_nativeStart(JNIEnv*, jobject) {
    if (!engine) engine = new AudioEngine();
    engine->start();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_assistivenavigation_AudioEngine_nativeStop(JNIEnv*, jobject) {
    if (engine) { engine->stop(); delete engine; engine = nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_assistivenavigation_AudioEngine_nativeSetSpatialParams(
        JNIEnv*, jobject, jfloat azimuth, jfloat distance) {
    if (engine) engine->setSpatialParams(azimuth, distance);
}