#include "AudioEngine.h"
#include <cmath>
#include <jni.h>

#define PI 3.14159265358979f

AudioEngine::AudioEngine() {}

AudioEngine::~AudioEngine() {
    stop();
}

bool AudioEngine::start() {

    oboe::AudioStreamBuilder builder;

    builder.setDirection(oboe::Direction::Output)
            ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
            ->setSharingMode(oboe::SharingMode::Shared)
            ->setFormat(oboe::AudioFormat::Float)
            ->setChannelCount(2)
            ->setBufferCapacityInFrames(1920)
            ->setCallback(this);

    if (builder.openStream(&stream) != oboe::Result::OK) {
        return false;
    }

    if (stream->requestStart() != oboe::Result::OK) {
        return false;
    }

    return true;
}

void AudioEngine::stop() {
    if (stream) {
        stream->close();
        stream = nullptr;
    }
}

void AudioEngine::setSpatialParams(float azimuthDeg, float distanceMeters) {
    azimuth.store(azimuthDeg);
    distance.store(distanceMeters);
}

void AudioEngine::computeStereoGains(float& left, float& right) {

    float az = azimuth.load();
    float dist = distance.load();

    // Clamp azimuth to [-90, 90]
    if (az < -90.0f) az = -90.0f;
    if (az > 90.0f)  az = 90.0f;

    float pan = az / 90.0f;  // -1 to +1

    // Equal power panning
    float angle = (pan + 1.0f) * PI / 4.0f;
    left  = cosf(angle);
    right = sinf(angle);

    // Distance attenuation
    float gain = 1.0f / (1.0f + dist);

    left  *= gain;
    right *= gain;
}

oboe::DataCallbackResult
AudioEngine::onAudioReady(oboe::AudioStream* stream,
                          void* audioData,
                          int32_t numFrames) {

    float* output = static_cast<float*>(audioData);

    float sampleRate = stream->getSampleRate();
    float freq = 440.0f;

    float leftGain, rightGain;
    computeStereoGains(leftGain, rightGain);

    for (int i = 0; i < numFrames; ++i) {

        float value = sinf(phase);
        phase += 2.0f * PI * freq / sampleRate;

        if (phase > 2.0f * PI) phase -= 2.0f * PI;

        output[i * 2]     = value * leftGain;
        output[i * 2 + 1] = value * rightGain;
    }

    return oboe::DataCallbackResult::Continue;
}

static AudioEngine* engine = nullptr;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_assistivenavigation_AudioEngine_nativeStart(
        JNIEnv*, jobject) {

if (!engine) {
engine = new AudioEngine();
}

engine->start();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_assistivenavigation_AudioEngine_nativeStop(
        JNIEnv*, jobject) {

if (engine) {
engine->stop();
delete engine;
engine = nullptr;
}
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_assistivenavigation_AudioEngine_nativeSetSpatialParams(
        JNIEnv*, jobject,
jfloat azimuth,
        jfloat distance) {

if (engine) {
engine->setSpatialParams(azimuth, distance);
}
}