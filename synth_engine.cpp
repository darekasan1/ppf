#include "synth_engine.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>

const double PI = 3.14159265358979323846;

// 乱数生成器 (スレッドセーフではないため、本来は各スレッドで個別に使用するのが望ましい)
static std::mt19937 gen(std::random_device{}());
static std::uniform_real_distribution<> dis(-0.5, 0.5);

double midi_to_hz(int note) {
    return 440.0 * std::pow(2.0, (static_cast<double>(note) - 69.0) / 12.0);
}

double poly_blep(double t, double dt) {
    if (t < dt) {
        t = t / dt;
        return t + t - t * t - 1.0;
    } else if (t > 1.0 - dt) {
        t = (t - 1.0) / dt;
        return t * t + t + t + 1.0;
    } else {
        return 0.0;
    }
}

// ドラム音源カーネル
void render_drum_loop(
    std::vector<double>& output_signal, const std::vector<double>& envelope,
    int note, int velocity, int oversampled_rate)
{
    double norm_vel = static_cast<double>(velocity) / 127.0;

    if (note == 35 || note == 36) { // Kick
        double freq = 150.0;
        double phase = 0.0;
        for (size_t i = 0; i < output_signal.size(); ++i) {
            double inc = (freq * std::exp(-static_cast<double>(i) * 25.0 / oversampled_rate)) / oversampled_rate;
            phase += inc;
            if (phase >= 1.0) phase -= 1.0;
            double vca_out = std::sin(phase * 2.0 * PI) * envelope[i];
            output_signal[i] = std::tanh(vca_out * 3.0 * norm_vel);
        }
    } else if (note == 38 || note == 40) { // Snare
        double freq = 180.0;
        double phase = 0.0;
        for (size_t i = 0; i < output_signal.size(); ++i) {
            double inc = freq / oversampled_rate;
            phase += inc;
            if (phase >= 1.0) phase -= 1.0;
            double noise = (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
            double vca_out = (std::sin(phase * 2.0 * PI) * 0.3 + noise * 0.7) * envelope[i];
            output_signal[i] = std::tanh(vca_out * 2.5 * norm_vel);
        }
    } else { // Hi-hats & Cymbals
        double alpha = 0.95, last_in = 0.0, last_out = 0.0;
        for (size_t i = 0; i < output_signal.size(); ++i) {
            double noise = (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
            double vcf_out = alpha * (last_out + noise - last_in);
            last_in = noise;
            last_out = vcf_out;
            double vca_out = vcf_out * envelope[i];
            output_signal[i] = std::tanh(vca_out * 1.5 * norm_vel);
        }
    }
}

// シンセ音源カーネル
void render_synth_loop(
    std::vector<double>& output_signal, const std::vector<double>& envelope,
    int velocity, const SynthConfig& config, int note, int oversampled_rate)
{
    double base_freq = midi_to_hz(note);
    double r1 = dis(gen);
    double r2 = dis(gen);
    double freq_a = base_freq + r1 * config.DRIFT_AMOUNT;
    double freq_b = freq_a * std::pow(2.0, config.DETUNE_CENTS / 1200.0) + r2 * config.DRIFT_AMOUNT;
    
    double phase_a = 0.0, phase_b = 0.0;
    double k = config.FILTER_RESONANCE_Q;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, feedback = 0.0;
    double lfo_phase = 0.0, lfo_inc = config.LFO_RATE / oversampled_rate;
    double norm_vel = static_cast<double>(velocity) / 127.0;

    for (size_t i = 0; i < output_signal.size(); ++i) {
        lfo_phase += lfo_inc;
        if (lfo_phase >= 1.0) lfo_phase -= 1.0;
        double lfo_out = std::sin(lfo_phase * 2.0 * PI);
        
        double inc_a = (freq_a * (1.0 + lfo_out * config.LFO_DEPTH)) / oversampled_rate;
        double inc_b = (freq_b * (1.0 + lfo_out * config.LFO_DEPTH)) / oversampled_rate;
        
        double current_cutoff = std::max(10.0, std::min(static_cast<double>(oversampled_rate) / 2.1, config.FILTER_CUTOFF + envelope[i] * config.FILTER_ENV_AMOUNT));
        double g = 1.0 - std::exp(-2.0 * PI * current_cutoff / oversampled_rate);
        
        phase_a += inc_a;
        double saw_a = (phase_a * 2.0 - 1.0) - poly_blep(phase_a, inc_a);
        if (phase_a >= 1.0) phase_a -= 1.0;
        
        phase_b += inc_b;
        double saw_b = (phase_b * 2.0 - 1.0) - poly_blep(phase_b, inc_b);
        if (phase_b >= 1.0) phase_b -= 1.0;
        
        double vco_out = saw_a + saw_b;
        
        double filter_in = std::tanh(vco_out - feedback * k);
        s1 += g * (filter_in - s1);
        s2 += g * (s1 - s2);
        s3 += g * (s2 - s3);
        s4 += g * (s3 - s4);
        double vcf_out = s4;
        feedback = s4;
        
        double vca_out = vcf_out * envelope[i] * norm_vel;
        output_signal[i] = std::tanh(vca_out * config.DRIVE);
    }
}


std::vector<float> render_voice(
    int note, int velocity, double duration, const SynthConfig& config,
    int sample_rate, int oversample_factor, bool is_drum)
{
    int oversampled_rate = sample_rate * oversample_factor;
    double total_duration_with_release = duration + config.RELEASE_TIME;
    size_t total_samples_oversampled = static_cast<size_t>(total_duration_with_release * oversampled_rate);

    if (total_samples_oversampled == 0) return {};

    std::vector<double> envelope(total_samples_oversampled, 0.0);
    size_t note_on_samples = static_cast<size_t>(duration * oversampled_rate);
    
    size_t attack_samples_full = static_cast<size_t>(config.ATTACK_TIME * oversampled_rate);
    size_t actual_attack_samples = std::min(attack_samples_full, note_on_samples);
    if (actual_attack_samples > 0) {
        for (size_t i = 0; i < actual_attack_samples; ++i) envelope[i] = static_cast<double>(i) / actual_attack_samples;
    }
    size_t samples_after_attack = note_on_samples - actual_attack_samples;
    if (samples_after_attack > 0) {
        size_t decay_samples_full = static_cast<size_t>(config.DECAY_TIME * oversampled_rate);
        size_t actual_decay_samples = std::min(decay_samples_full, samples_after_attack);
        if (actual_decay_samples > 0) {
            for (size_t i = 0; i < actual_decay_samples; ++i) envelope[actual_attack_samples + i] = 1.0 - (1.0 - config.SUSTAIN_LEVEL) * (static_cast<double>(i) / actual_decay_samples);
        }
        size_t sustain_samples = samples_after_attack - actual_decay_samples;
        if (sustain_samples > 0) {
            for (size_t i = 0; i < sustain_samples; ++i) envelope[actual_attack_samples + actual_decay_samples + i] = config.SUSTAIN_LEVEL;
        }
    }
    double release_start_level = config.SUSTAIN_LEVEL;
    if (note_on_samples > 0 && note_on_samples <= total_samples_oversampled) release_start_level = envelope[note_on_samples - 1];
    else if (note_on_samples == 0) release_start_level = 0.0;
    size_t release_samples = static_cast<size_t>(config.RELEASE_TIME * oversampled_rate);
    if (release_samples > 0 && note_on_samples < envelope.size()) {
        size_t actual_release_samples = std::min(release_samples, total_samples_oversampled - note_on_samples);
        for (size_t i = 0; i < actual_release_samples; ++i) envelope[note_on_samples + i] = release_start_level * (1.0 - static_cast<double>(i) / actual_release_samples);
    }
    
    std::vector<double> output_signal(total_samples_oversampled);

    if (is_drum) {
        render_drum_loop(output_signal, envelope, note, velocity, oversampled_rate);
    } else {
        render_synth_loop(output_signal, envelope, velocity, config, note, oversampled_rate);
    }
    
    std::vector<float> downsampled;
    downsampled.reserve(output_signal.size() / oversample_factor);
    for (size_t i = 0; i < output_signal.size(); i += oversample_factor) {
        downsampled.push_back(static_cast<float>(output_signal[i]));
    }
    
    return downsampled;
}
