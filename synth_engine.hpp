#pragma once
#include <vector>

struct SynthConfig {
    double DETUNE_CENTS, DRIFT_AMOUNT, FILTER_CUTOFF, FILTER_RESONANCE_Q,
           FILTER_ENV_AMOUNT, ATTACK_TIME, DECAY_TIME, SUSTAIN_LEVEL,
           RELEASE_TIME, DRIVE, LFO_RATE, LFO_DEPTH;
};

// ★★★【修正点】最後の time_offset_in_note 引数を削除 ★★★
std::vector<float> render_voice(
    int note, int velocity, double duration, const SynthConfig& config,
    int sample_rate, int oversample_factor, bool is_drum
);
