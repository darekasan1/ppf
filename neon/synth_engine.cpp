#include "synth_engine.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <arm_neon.h>  // NEON intrinsics

const double PI = 3.14159265358979323846;

static thread_local std::mt19937 gen(std::random_device{}());
static thread_local std::uniform_real_distribution<> dis(-0.5, 0.5);

double midi_to_hz(int note) {
    return 440.0 * std::pow(2.0, (static_cast<double>(note) - 69.0) / 12.0);
}

// PolyBLEP最適化：分岐削減
inline double poly_blep(double t, double dt) {
    // branchless version
    double a = t / dt;
    double b = (t - 1.0) / dt;

    double case1 = a + a - a * a - 1.0;
    double case2 = b * b + b + b + 1.0;

    return (t < dt) * case1 + (t > 1.0 - dt) * case2;
}

// NEON最適化版：エンベロープ処理
void apply_envelope_neon(std::vector<double>& output, const std::vector<double>& envelope) {
    size_t i = 0;
    size_t len = output.size();

    // 2サンプル同時処理（float64x2_t）
    for (; i + 2 <= len; i += 2) {
        float64x2_t out_vec = vld1q_f64(&output[i]);
        float64x2_t env_vec = vld1q_f64(&envelope[i]);
        float64x2_t result = vmulq_f64(out_vec, env_vec);
        vst1q_f64(&output[i], result);
    }

    // 残り
    for (; i < len; ++i) {
        output[i] *= envelope[i];
    }
}

// 高速tanh近似（NEON用）
inline float64x2_t fast_tanh_neon(float64x2_t x) {
    // tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
    float64x2_t x2 = vmulq_f64(x, x);
    float64x2_t c27 = vdupq_n_f64(27.0);
    float64x2_t c9 = vdupq_n_f64(9.0);

    float64x2_t num = vmlaq_f64(c27, x2, vdupq_n_f64(1.0)); // 27 + x^2
    float64x2_t den = vmlaq_f64(c27, x2, c9); // 27 + 9*x^2

    // 除算は遅いので逆数推定
    float64x2_t inv_den = vrecpeq_f64(den);
    inv_den = vmulq_f64(vrecpsq_f64(den, inv_den), inv_den); // Newton-Raphson

    float64x2_t result = vmulq_f64(vmulq_f64(x, num), inv_den);
    return result;
}

// NEON最適化：ドラム音源
void render_drum_loop_neon(
    std::vector<double>& output_signal, const std::vector<double>& envelope,
    int note, int velocity, int oversampled_rate)
{
    double norm_vel = static_cast<double>(velocity) / 127.0;
    size_t len = output_signal.size();

    if (note == 35 || note == 36) { // Kick
        double freq = 150.0;
        double phase = 0.0;
        double decay_factor = 25.0 / oversampled_rate;

        // スカラー処理（位相管理が必要）
        for (size_t i = 0; i < len; ++i) {
            double inc = (freq * std::exp(-static_cast<double>(i) * decay_factor)) / oversampled_rate;
            phase += inc;
            if (phase >= 1.0) phase -= 1.0;
            output_signal[i] = std::sin(phase * 2.0 * PI) * envelope[i] * 3.0 * norm_vel;
        }

        // tanh処理をNEONで
        size_t i = 0;
        for (; i + 2 <= len; i += 2) {
            float64x2_t val = vld1q_f64(&output_signal[i]);
            float64x2_t result = fast_tanh_neon(val);
            vst1q_f64(&output_signal[i], result);
        }
        for (; i < len; ++i) {
            output_signal[i] = std::tanh(output_signal[i]);
        }

    } else if (note == 38 || note == 40) { // Snare
        double freq = 180.0;
        double phase = 0.0;
        double inv_rate = 1.0 / oversampled_rate;

        // ノイズ事前生成
        std::vector<double> noise_buffer(len);
        for (size_t i = 0; i < len; ++i) {
            noise_buffer[i] = (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
        }

        for (size_t i = 0; i < len; ++i) {
            phase += freq * inv_rate;
            if (phase >= 1.0) phase -= 1.0;
            output_signal[i] = (std::sin(phase * 2.0 * PI) * 0.3 + noise_buffer[i] * 0.7) * envelope[i] * 2.5 * norm_vel;
        }

        // tanh NEON
        size_t i = 0;
        for (; i + 2 <= len; i += 2) {
            float64x2_t val = vld1q_f64(&output_signal[i]);
            float64x2_t result = fast_tanh_neon(val);
            vst1q_f64(&output_signal[i], result);
        }
        for (; i < len; ++i) {
            output_signal[i] = std::tanh(output_signal[i]);
        }

    } else { // Hi-hats
        double alpha = 0.95, last_in = 0.0, last_out = 0.0;
        for (size_t i = 0; i < len; ++i) {
            double noise = (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
            double vcf_out = alpha * (last_out + noise - last_in);
            last_in = noise;
            last_out = vcf_out;
            output_signal[i] = std::tanh(vcf_out * envelope[i] * 1.5 * norm_vel);
        }
    }
}

// シンセ音源（NEON最適化）
void render_synth_loop_neon(
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
    double lfo_phase = 0.0;
    double lfo_inc = config.LFO_RATE / oversampled_rate;
    double norm_vel = static_cast<double>(velocity) / 127.0;

    // ループアンローリング：2サンプルずつ処理
    size_t i = 0;
    size_t len = output_signal.size();

    for (; i < len; ++i) {
        lfo_phase += lfo_inc;
        if (lfo_phase >= 1.0) lfo_phase -= 1.0;
        double lfo_out = std::sin(lfo_phase * 2.0 * PI);

        double inc_a = (freq_a * (1.0 + lfo_out * config.LFO_DEPTH)) / oversampled_rate;
        double inc_b = (freq_b * (1.0 + lfo_out * config.LFO_DEPTH)) / oversampled_rate;

        double current_cutoff = std::max(10.0, std::min(static_cast<double>(oversampled_rate) / 2.1,
                                        config.FILTER_CUTOFF + envelope[i] * config.FILTER_ENV_AMOUNT));
        double g = 1.0 - std::exp(-2.0 * PI * current_cutoff / oversampled_rate);

        phase_a += inc_a;
        double saw_a = (phase_a * 2.0 - 1.0) - poly_blep(phase_a, inc_a);
        if (phase_a >= 1.0) phase_a -= 1.0;

        phase_b += inc_b;
        double saw_b = (phase_b * 2.0 - 1.0) - poly_blep(phase_b, inc_b);
        if (phase_b >= 1.0) phase_b -= 1.0;

        double vco_out = saw_a + saw_b;
        double filter_in = std::tanh(vco_out - feedback * k);

        // Ladder filter（手動最適化）
        s1 += g * (filter_in - s1);
        s2 += g * (s1 - s2);
        s3 += g * (s2 - s3);
        s4 += g * (s3 - s4);
        feedback = s4;

        output_signal[i] = s4 * envelope[i] * norm_vel * config.DRIVE;
    }

    // 最終tanh処理をNEONで
    i = 0;
    for (; i + 2 <= len; i += 2) {
        float64x2_t val = vld1q_f64(&output_signal[i]);
        float64x2_t result = fast_tanh_neon(val);
        vst1q_f64(&output_signal[i], result);
    }
    for (; i < len; ++i) {
        output_signal[i] = std::tanh(output_signal[i]);
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

    // エンベロープ生成最適化
    std::vector<double> envelope;
    envelope.reserve(total_samples_oversampled);
    size_t note_on_samples = static_cast<size_t>(duration * oversampled_rate);

    size_t attack_samples = std::min(static_cast<size_t>(config.ATTACK_TIME * oversampled_rate), note_on_samples);
    double attack_inc = attack_samples > 0 ? 1.0 / attack_samples : 0.0;
    for (size_t i = 0; i < attack_samples; ++i) {
        envelope.push_back(i * attack_inc);
    }

    size_t samples_after_attack = note_on_samples - attack_samples;
    size_t decay_samples = std::min(static_cast<size_t>(config.DECAY_TIME * oversampled_rate), samples_after_attack);
    double decay_range = 1.0 - config.SUSTAIN_LEVEL;
    double decay_inc = decay_samples > 0 ? decay_range / decay_samples : 0.0;
    for (size_t i = 0; i < decay_samples; ++i) {
        envelope.push_back(1.0 - i * decay_inc);
    }

    size_t sustain_samples = samples_after_attack - decay_samples;
    for (size_t i = 0; i < sustain_samples; ++i) {
        envelope.push_back(config.SUSTAIN_LEVEL);
    }

    double release_start = note_on_samples > 0 ? envelope.back() : 0.0;
    size_t release_samples = static_cast<size_t>(config.RELEASE_TIME * oversampled_rate);
    double release_inc = release_samples > 0 ? release_start / release_samples : 0.0;
    for (size_t i = 0; i < release_samples; ++i) {
        envelope.push_back(release_start - i * release_inc);
    }

    std::vector<double> output_signal(envelope.size());

    if (is_drum) {
        render_drum_loop_neon(output_signal, envelope, note, velocity, oversampled_rate);
    } else {
        render_synth_loop_neon(output_signal, envelope, velocity, config, note, oversampled_rate);
    }

    // ダウンサンプリング（NEON最適化可能だが、シンプルに）
    std::vector<float> downsampled;
    downsampled.reserve(output_signal.size() / oversample_factor + 1);
    for (size_t i = 0; i < output_signal.size(); i += oversample_factor) {
        downsampled.push_back(static_cast<float>(output_signal[i]));
    }

    return downsampled;
}
