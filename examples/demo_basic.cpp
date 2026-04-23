/**
 * @file demo_basic.cpp
 * @brief Basic demo of Cutie-CPP video object segmentation.
 *
 * Usage:
 *   ./demo_basic --video FILE --mask FILE [--model-dir DIR] [--visualize] [--frame-skip N]
 *
 * Arguments:
 *   --video FILE     - Input video file
 *   --mask  FILE     - PNG mask for the first frame (pixel values = object IDs)
 *   --model-dir DIR  - Optional. Directory with 6 ONNX submodule files;
 *                      auto-detected from build/install paths if omitted.
 *   --visualize      - Optional. Show preview window in addition to saving frames.
 *   --frame-skip N   - Optional. Skip N frames between inferences
 *                      (0 = every frame, 1 = every 2nd, ...). Default 0.
 */

#include "cutie/cutie.h"

#include <argparse/argparse.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <linden_logger/logger_interface.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
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
    // Resolve output directory: <executable dir>/output
    namespace fs = std::filesystem;
    fs::path exe_dir;
    {
        std::error_code ec;
        fs::path exe = fs::canonical("/proc/self/exe", ec);
        exe_dir = !ec ? exe.parent_path() : fs::absolute(fs::path(argv[0])).parent_path();
    }
    fs::path output_dir = exe_dir / "output";
    {
        std::error_code ec;
        fs::remove_all(output_dir, ec);
        fs::create_directories(output_dir, ec);
        if (ec) {
            std::cerr << "Failed to create output directory " << output_dir
                      << ": " << ec.message() << std::endl;
            return 1;
        }
    }

    // 初始化 logger（同时输出到终端和文件）
    linden::log::StdLogger::FileLogConfig log_config(
        output_dir.string(), 7, 0, 0, "cutie_debug");
    auto logger = linden::log::StdLogger::instance(log_config);
    logger->set_level(linden::log::LogLevel::DEBUG);

    argparse::ArgumentParser program("demo_basic");
    program.add_description("Cutie-CPP video object segmentation demo.");

    program.add_argument("--video", "-i")
        .required()
        .help("Input video file");
    program.add_argument("--mask", "-k")
        .required()
        .help("First-frame PNG mask (pixel values = object IDs)");
    program.add_argument("--model-dir", "-m")
        .default_value(std::string{})
        .help("Directory with 6 ONNX submodule files (auto-detected if omitted)");
    program.add_argument("--visualize", "-v")
        .default_value(false)
        .implicit_value(true)
        .help("Show preview window in addition to saving frames");
    program.add_argument("--frame-skip", "-s")
        .default_value(0)
        .scan<'i', int>()
        .help("Skip N frames between inferences (0 = process every frame, 1 = every 2nd, ...)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& e) {
        logger->error("{}\n\n{}", e.what(), program.help().str());
        return 1;
    }

    const bool show_window = program.get<bool>("--visualize");
    const std::string video_path = program.get<std::string>("--video");
    const std::string mask_path  = program.get<std::string>("--mask");
    std::string model_dir        = program.get<std::string>("--model-dir");
    const int frame_skip         = std::max(0, program.get<int>("--frame-skip"));

    if (model_dir.empty()) {
        model_dir = find_default_onnx_dir();
        if (model_dir.empty()) {
            logger->error("Error: could not find ONNX model files.");
            logger->error("  Tried build path  : {}",
#ifdef CUTIE_BUILD_MODEL_DIR
                CUTIE_BUILD_MODEL_DIR
#else
                "(not configured)"
#endif
            );
            logger->error("  Tried install path: {}",
#ifdef CUTIE_INSTALL_MODEL_DIR
                CUTIE_INSTALL_MODEL_DIR
#else
                "(not configured)"
#endif
            );
            logger->error("Run 'cmake --build . --target download_models' and export ONNX files,");
            logger->error("or pass --model-dir explicitly.");
            return 1;
        }
        logger->info("Auto-detected model directory: {}", model_dir);
    }

    logger->info("Saving results to: {}", output_dir.string());

    // 1. Create config
    auto config = CutieConfig::base_default(model_dir);
    config.max_internal_size = 320;
    config.mem_every = 5;
    config.model_prefix = find_model_prefix(model_dir);
    if (config.model_prefix.empty()) {
        logger->error("Error: no prefixed ONNX files found in {}", model_dir);
        logger->error("  Expected pattern: <prefix>_pixel_encoder.onnx");
        return 1;
    }
    logger->info("Using model prefix: {}", config.model_prefix);

    // 2. Create processor
    logger->info("Loading model from {}...", model_dir);
    CutieProcessor processor(config);

    // 3. Open video
    ::cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        logger->error("Failed to open video: {}", video_path);
        return 1;
    }
    {
        const double fps    = cap.get(::cv::CAP_PROP_FPS);
        const double nframe = cap.get(::cv::CAP_PROP_FRAME_COUNT);
        logger->info("Video: {}  fps={}  reported_frame_count={}", video_path, fps, nframe);
    }

    // 4. Read first-frame mask
    ::cv::Mat mask_img = ::cv::imread(mask_path, ::cv::IMREAD_GRAYSCALE);
    if (mask_img.empty()) {
        logger->error("Failed to read mask: {}", mask_path);
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
    logger->info("Found {} objects in mask", objects.size());

    // ---- Background save thread + queue ------------------------------------
    struct SaveJob {
        int idx;
        ::cv::Mat image;  // BGR; owns its own buffer
    };
    std::queue<SaveJob> save_queue;
    std::mutex queue_mu;
    std::condition_variable queue_cv;
    std::atomic<bool> producer_done{false};
    constexpr size_t kMaxQueue = 32;  // back-pressure to bound memory

    std::thread saver([&]() {
        while (true) {
            SaveJob job;
            {
                std::unique_lock<std::mutex> lk(queue_mu);
                queue_cv.wait(lk, [&] { return !save_queue.empty() || producer_done.load(); });
                if (save_queue.empty() && producer_done.load()) break;
                job = std::move(save_queue.front());
                save_queue.pop();
            }
            queue_cv.notify_all();  // wake producer if it was blocked on capacity

            std::ostringstream oss;
            oss << "frame_" << std::setw(6) << std::setfill('0') << job.idx << ".jpg";
            fs::path out_path = output_dir / oss.str();
            if (!::cv::imwrite(out_path.string(), job.image)) {
                logger->error("Failed to write {}", out_path.string());
            }
        }
    });

    logger->info("=== Cutie-CPP Debug Session Started ===");
    logger->info("Video: {}", video_path);
    logger->info("Mask: {}", mask_path);
    logger->info("Model: {}", config.model_prefix);
    logger->info("Frame skip: {}", frame_skip);

    // 5. Process video frame by frame
    ::cv::Mat frame;
    int frame_idx = 0;        // real (0-based) source frame index
    int processed_cnt = 0;    // number of frames actually inferred & saved
    bool user_quit = false;

    while (cap.read(frame)) {
        // Some codecs return true with an empty frame at EOF — treat as end.
        if (frame.empty()) break;

        // Skip every (frame_skip) frames between inferences. Frame 0 always runs to init.
        const int stride = frame_skip + 1;
        if (frame_idx != 0 && (frame_idx % stride) != 0) {
            ++frame_idx;
            continue;
        }

        cutie::types::CutieMask result;

        auto t0 = std::chrono::steady_clock::now();

        if (frame_idx == 0) {
            result = processor.step(frame, idx_mask, objects);
            logger->debug("[demo] frame {:4d} | init | objects={}",
                          frame_idx, result.object_ids.size());
        } else {
            result = processor.step(frame);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 统计每个目标的像素数，方便排查分割结果
        int total_fg = 0;
        for (auto obj_id : result.object_ids) {
            int px_count = ::cv::countNonZero(result.index_mask == obj_id);
            total_fg += px_count;
            logger->debug("[demo] frame {:4d} | obj {} pixels={}", frame_idx, obj_id, px_count);
        }
        int total_px = result.index_mask.rows * result.index_mask.cols;
        logger->debug("[demo] frame {:4d} | {:.1f} ms | fg={}/{} ({:.1f}%)",
                      frame_idx, ms, total_fg, total_px,
                      total_px > 0 ? 100.0 * total_fg / total_px : 0.0);

        ::cv::Mat vis = visualize(frame, result.index_mask);

        // Optional clone for display before we move into the save queue.
        ::cv::Mat vis_for_show;
        if (show_window) {
            vis_for_show = vis.clone();
        }

        // Enqueue for background save (block if queue too large to bound memory).
        // Filename uses the REAL source frame index, so e.g. with --frame-skip 2
        // you'll see frame_000000.jpg, frame_000003.jpg, frame_000006.jpg ...
        {
            std::unique_lock<std::mutex> lk(queue_mu);
            queue_cv.wait(lk, [&] { return save_queue.size() < kMaxQueue; });
            save_queue.push(SaveJob{frame_idx, std::move(vis)});
        }
        queue_cv.notify_one();
        ++processed_cnt;

        if (show_window) {
            ::cv::imshow("Cutie-CPP Demo", vis_for_show);
            int key = ::cv::waitKey(1);
            if (key == 27 || key == 'q') {
                user_quit = true;
                break;
            }
        }

        ++frame_idx;
    }

    // Signal saver to drain and exit.
    producer_done.store(true);
    queue_cv.notify_all();
    saver.join();

    if (show_window) {
        ::cv::destroyAllWindows();
    }
    logger->info("Done. read {} source frames, saved {} inferred frames to {}",
                 frame_idx, processed_cnt, output_dir.string());
    logger->debug("[demo] done | source_frames={} | saved={}", frame_idx, processed_cnt);
    (void)user_quit;
    return 0;
}
