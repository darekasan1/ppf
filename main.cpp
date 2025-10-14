#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <numeric>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "MidiFile.h"
#include "synth_engine.hpp"

// (グローバル設定、NoteInfo構造体、WAV書き出し関数、ワーカー関数は変更なし)
#pragma region Unchanged
const int OVERSAMPLE_FACTOR = 2;
const int SAMPLE_RATE = 48000;
const SynthConfig SYNTH_CONFIG = {
    8.0, 0.05, 1000.0, 3.5, 6000.0, 0.05, 0.4, 0.7, 0.5, 1.5, 5.0, 0.01
};
const SynthConfig DRUM_CONFIG = {
    0.0, 0.0, 8000.0, 0.5, 0.0, 0.001, 0.05, 0.3, 0.1, 1.2, 0.0, 0.0
};
const float SYNTH_MASTER_VOLUME = 1.2f;
const float DRUM_MASTER_VOLUME = 0.8f;

struct NoteInfo {
    int pitch, velocity;
    double start_time, duration;
    bool is_drum;
};

void saveWavFile(const std::string& filename, const std::vector<float>& audio_data, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    file.write("RIFF", 4);
    uint32_t chunk_size = 36 + audio_data.size() * sizeof(int16_t);
    file.write(reinterpret_cast<const char*>(&chunk_size), 4);
    file.write("WAVE", 4); file.write("fmt ", 4);
    uint32_t subchunk1_size = 16;
    file.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
    uint16_t audio_format = 1; file.write(reinterpret_cast<const char*>(&audio_format), 2);
    uint16_t num_channels = 1; file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    uint32_t byte_rate = sample_rate * num_channels * sizeof(int16_t);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    uint16_t block_align = num_channels * sizeof(int16_t);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    uint16_t bits_per_sample = 16; file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    file.write("data", 4);
    uint32_t subchunk2_size = audio_data.size() * sizeof(int16_t);
    file.write(reinterpret_cast<const char*>(&subchunk2_size), 4);
    std::vector<int16_t> int_data(audio_data.size());
    for(size_t i = 0; i < audio_data.size(); ++i) {
        int_data[i] = static_cast<int16_t>(audio_data[i] * 32767.0f);
    }
    file.write(reinterpret_cast<const char*>(int_data.data()), subchunk2_size);
}

std::pair<std::vector<float>, std::vector<float>> worker_function(const std::vector<NoteInfo>* notes, size_t start_index, size_t end_index, double total_duration) {
    std::vector<float> local_synth(static_cast<size_t>(total_duration * SAMPLE_RATE) + SAMPLE_RATE * 3, 0.0f);
    std::vector<float> local_drums(static_cast<size_t>(total_duration * SAMPLE_RATE) + SAMPLE_RATE * 3, 0.0f);
    for (size_t i = start_index; i < end_index; ++i) {
        const auto& note = (*notes)[i];
        const auto& config = note.is_drum ? DRUM_CONFIG : SYNTH_CONFIG;
        auto waveform = render_voice(note.pitch, note.velocity, note.duration,
                                     config, SAMPLE_RATE, OVERSAMPLE_FACTOR, note.is_drum);
        size_t start_sample = static_cast<size_t>(note.start_time * SAMPLE_RATE);
        auto& master_buffer = note.is_drum ? local_drums : local_synth;
        for (size_t j = 0; j < waveform.size(); ++j) {
            if (start_sample + j < master_buffer.size()) {
                master_buffer[start_sample + j] += waveform[j];
            }
        }
    }
    return {local_synth, local_drums};
}
#pragma endregion

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <midi_file> [-o <output.wav>]" << std::endl;
        return 1;
    }
    std::string midi_filename = argv[1];
    std::string output_filename = "";
    if (argc > 3 && std::string(argv[2]) == "-o") {
        output_filename = argv[3];
    }

    smf::MidiFile midi_file;
    if (!midi_file.read(midi_filename)) {
        std::cerr << "Error reading MIDI file: " << midi_filename << std::endl;
        return 1;
    }
    midi_file.doTimeAnalysis();
    midi_file.linkNotePairs();

    std::vector<NoteInfo> notes_to_render;
    for (int track = 0; track < midi_file.getTrackCount(); ++track) {
        for (int event = 0; event < midi_file[track].size(); ++event) {
            if (midi_file[track][event].isNoteOn()) {
                
                // ★★★【修正点】チャンネル番号でドラムを判定 ★★★
                bool is_drum_event = (midi_file[track][event].getChannel() == 9);
                
                double duration = midi_file[track][event].getDurationInSeconds();
                
                if (is_drum_event) {
                    if (midi_file[track][event].getKeyNumber() > 49) duration = 0.8;
                    else if (midi_file[track][event].getKeyNumber() == 46) duration = 0.3;
                    else duration = 0.15;
                }

                notes_to_render.push_back({
                    midi_file[track][event].getKeyNumber(),
                    midi_file[track][event].getVelocity(),
                    midi_file[track][event].seconds,
                    duration,
                    is_drum_event
                });
            }
        }
    }
    
    // (以降の並列処理、ミックスダウン、ファイル書き出しは変更なし)
    #pragma region Finalize
    double total_duration = midi_file.getFileDurationInSeconds();
    auto start_time = std::chrono::high_resolution_clock::now();
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "Rendering " << notes_to_render.size() << " notes with " << num_threads << " threads..." << std::endl;
    std::vector<std::future<std::pair<std::vector<float>, std::vector<float>>>> futures;
    size_t notes_per_thread = (notes_to_render.size() + num_threads - 1) / num_threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t start_index = i * notes_per_thread;
        size_t end_index = std::min(start_index + notes_per_thread, notes_to_render.size());
        if (start_index < end_index) {
            futures.push_back(std::async(std::launch::async, worker_function, &notes_to_render, start_index, end_index, total_duration));
        }
    }
    
    std::vector<float> master_synth(static_cast<size_t>(total_duration * SAMPLE_RATE) + SAMPLE_RATE * 3, 0.0f);
    std::vector<float> master_drums(static_cast<size_t>(total_duration * SAMPLE_RATE) + SAMPLE_RATE * 3, 0.0f);
    for (auto& fut : futures) {
        auto result = fut.get();
        for(size_t i = 0; i < result.first.size(); ++i) master_synth[i] += result.first[i];
        for(size_t i = 0; i < result.second.size(); ++i) master_drums[i] += result.second[i];
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Rendering finished in " << elapsed.count() << " seconds." << std::endl;
    
    std::vector<float> final_output(master_synth.size());
    float peak = 0.0f;
    for (size_t i = 0; i < final_output.size(); ++i) {
        final_output[i] = master_synth[i] * SYNTH_MASTER_VOLUME + master_drums[i] * DRUM_MASTER_VOLUME;
        if (std::abs(final_output[i]) > peak) {
            peak = std::abs(final_output[i]);
        }
    }
    if (peak > 0.0f) {
        for (float& sample : final_output) {
            sample = (sample / peak) * 0.8f;
        }
    }
    
    if (!output_filename.empty()) {
        std::cout << "Saving to " << output_filename << "..." << std::endl;
        saveWavFile(output_filename, final_output, SAMPLE_RATE);
    }
    #pragma endregion

    return 0;
}
