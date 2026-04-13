/**
 * @file demo_basic.cpp
 * @brief Basic demo of Cutie-CPP video object segmentation.
 *
 * Usage:
 *   ./demo_basic <video_path> <first_frame_mask>
 *   ./demo_basic <model_dir> <video_path> <first_frame_mask>
 *
 * Arguments:
 *   model_dir        - (optional) Directory with 6 ONNX submodule files.
 *                      If omitted, auto-detected from build / install paths.
 *   video_path       - Input video file
 *   first_frame_mask - PNG mask for the first frame (pixel values = object IDs)
 */

#include "cutie/cutie.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <linden_logger/logger_interface.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// ---------------------------------------------------------------------------
// Auto-detect a model directory that contains the ONNX submodule files.
// CMake embeds CUTIE_BUILD_MODEL_DIR and CUTIE_INSTALL_MODEL_DIR at compile
// time via target_compile_definitions.
// ---------------------------------------------------------------------------
static std::string find_default_onnx_dir() {
    // Sentinel file that must be present for a valid model directory
    const std::string sentinel = "pixel_encoder.onnx";

#ifdef CUTIE_BUILD_MODEL_DIR
    {
        std::filesystem::path p(CUTIE_BUILD_MODEL_DIR);
        if (std::filesystem::exists(p / sentinel)) {
            return p.string();
        }
    }
#endif

#ifdef CUTIE_INSTALL_MODEL_DIR
    {
        std::filesystem::path p(CUTIE_INSTALL_MODEL_DIR);
        if (std::filesystem::exists(p / sentinel)) {
            return p.string();
        }
    }
#endif

    return "";
}

// Scan model_dir for a file matching "*_pixel_encoder.onnx" and extract the prefix.
static std::string find_model_prefix(const std::string& model_dir) {
    namespace fs = std::filesystem;
    const std::string suffix = "_pixel_encoder.onnx";
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        const std::string fname = entry.path().filename().string();
        if (fname.size() > suffix.size() &&
            fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return fname.substr(0, fname.size() - suffix.size());
        }
    }
    return "";
}

using namespace cutie::cv::segmentation;

// Color palette for visualization
static cv::Vec3b id_to_color(int id) {
    static const cv::Vec3b palette[] = {
        {0, 0, 0},       // 0: background
        {255, 0, 0},     // 1: blue
        {0, 255, 0},     // 2: green
        {0, 0, 255},     // 3: red
        {255, 255, 0},   // 4: cyan
        {255, 0, 255},   // 5: magenta
        {0, 255, 255},   // 6: yellow
        {128, 128, 255}, // 7
        {128, 255, 128}, // 8
    };
    if (id < 0 || id >= 9) return {128, 128, 128};
    return palette[id];
}

static cv::Mat visualize(const cv::Mat& frame, const cv::Mat& mask, float alpha = 0.4f) {
    cv::Mat vis = frame.clone();
    // Green semi-transparent overlay for all tracked objects (BGR: 0, 255, 0)
    static const cv::Vec3b kGreen = {0, 255, 0};
    for (int r = 0; r < mask.rows; ++r) {
        const int32_t* mp = mask.ptr<int32_t>(r);
        cv::Vec3b* vp = vis.ptr<cv::Vec3b>(r);
        for (int c = 0; c < mask.cols; ++c) {
            if (mp[c] > 0) {
                vp[c] = cv::Vec3b(
                    static_cast<uint8_t>(vp[c][0] * (1 - alpha) + kGreen[0] * alpha),
                    static_cast<uint8_t>(vp[c][1] * (1 - alpha) + kGreen[1] * alpha),
                    static_cast<uint8_t>(vp[c][2] * (1 - alpha) + kGreen[2] * alpha));
            }
        }
    }
    return vis;
}

int main(int argc, char** argv) {
    // Support: demo <video> <mask>  or  demo <model_dir> <video> <mask>
    if (argc < 3 || argc > 4) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0] << " <video_path> <first_frame_mask>\n"
            << "  " << argv[0] << " <model_dir> <video_path> <first_frame_mask>\n";
        return 1;
    }

    std::string model_dir;
    std::string video_path;
    std::string mask_path;

    if (argc == 4) {
        model_dir  = argv[1];
        video_path = argv[2];
        mask_path  = argv[3];
    } else {
        // argc == 3 — auto-detect model dir
        video_path = argv[1];
        mask_path  = argv[2];
        model_dir  = find_default_onnx_dir();
        if (model_dir.empty()) {
            std::cerr
                << "Error: could not find ONNX model files.\n"
                << "  Tried build path  : "
#ifdef CUTIE_BUILD_MODEL_DIR
                << CUTIE_BUILD_MODEL_DIR
#else
                << "(not configured)"
#endif
                << "\n  Tried install path: "
#ifdef CUTIE_INSTALL_MODEL_DIR
                << CUTIE_INSTALL_MODEL_DIR
#else
                << "(not configured)"
#endif
                << "\nRun 'cmake --build . --target download_models' and export ONNX files,\n"
                << "or pass <model_dir> explicitly.\n";
            return 1;
        }
        std::cout << "Auto-detected model directory: " << model_dir << std::endl;
    }

    // 1. Create config
    auto config = CutieConfig::base_default(model_dir);
    config.max_internal_size = 320;
    config.mem_every = 5;
    config.model_prefix = find_model_prefix(model_dir);
    if (config.model_prefix.empty()) {
        std::cerr << "Error: no prefixed ONNX files found in " << model_dir
                  << "\n  Expected pattern: <prefix>_pixel_encoder.onnx\n";
        return 1;
    }
    std::cout << "Using model prefix: " << config.model_prefix << std::endl;

    // 2. Create processor
    std::cout << "Loading model from " << model_dir << "..." << std::endl;
    CutieProcessor processor(config);

    // 3. Open video
    ::cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return 1;
    }

    // 4. Read first-frame mask
    ::cv::Mat mask_img = ::cv::imread(mask_path, ::cv::IMREAD_GRAYSCALE);
    if (mask_img.empty()) {
        std::cerr << "Failed to read mask: " << mask_path << std::endl;
        return 1;
    }
    ::cv::Mat idx_mask;
    mask_img.convertTo(idx_mask, CV_32SC1);

    // Discover object IDs from mask
    double min_v, max_v;
    ::cv::minMaxLoc(idx_mask, &min_v, &max_v);
    std::vector<cutie::ObjectId> objects;
    for (int id = 1; id <= static_cast<int>(max_v); ++id) {
        if (::cv::countNonZero(idx_mask == id) > 0) {
            objects.push_back(id);
        }
    }
    std::cout << "Found " << objects.size() << " objects in mask" << std::endl;

    // 5. Process video frame by frame
    ::cv::Mat frame;
    int frame_idx = 0;
    auto logger = linden::log::StdLogger::instance();

    while (cap.read(frame)) {
        cutie::types::CutieMask result;

        auto t0 = std::chrono::steady_clock::now();

        if (frame_idx == 0) {
            // First frame: provide mask
            result = processor.step(frame, idx_mask, objects);
            logger->debug("[demo] frame {:4d} | init | objects={}",
                          frame_idx, result.object_ids.size());
        } else {
            // Subsequent frames: propagate
            result = processor.step(frame);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        logger->debug("[demo] frame {:4d} | inference={:.1f} ms | objects={}",
                      frame_idx, ms, result.object_ids.size());

        // Visualize
        ::cv::Mat vis = visualize(frame, result.index_mask);
        ::cv::imshow("Cutie-CPP Demo", vis);

        int key = ::cv::waitKey(1);
        if (key == 27 || key == 'q') break;

        frame_idx++;
    }

    logger->debug("[demo] done | total frames={}", frame_idx);
    return 0;
}
