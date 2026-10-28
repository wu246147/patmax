#include "ShapeMatcher.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <tuple>

/**
 * @brief 验证真实 rings 图像定位，以及合成角度和尺度变化后的定位。
 * @param argc 命令行参数个数，程序名和图像目录共需两个。
 * @param argv argv[1] 为 HALCON rings 图像目录。
 * @return 0 表示全部检查通过，1 表示匹配失败，2 表示输入图像不可用。
 */
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: patmax_smoke <rings-directory>\n";
        return 2;
    }
    const std::string base = argv[1];
    const auto modelImage = cv::imread(base + "/rings_01.png", cv::IMREAD_GRAYSCALE);
    const auto sceneImage = cv::imread(base + "/rings_06.png", cv::IMREAD_GRAYSCALE);
    if (modelImage.empty() || sceneImage.empty()) {
        std::cerr << "Cannot load rings_01.png or rings_06.png\n";
        return 2;
    }
    patmax::Options options;
    options.minScore = 0.58f;
    options.maxMatches = 12;
    patmax::ShapeMatcher matcher;
    std::string error;
    if (!matcher.create(modelImage, cv::Rect(330, 110, 210, 210), options, error)) {
        std::cerr << "Model creation failed: " << error << '\n';
        return 1;
    }
    patmax::SearchStats stats;
    const auto matches = matcher.find(sceneImage, options, error, &stats);
    if (!error.empty()) {
        std::cerr << "Search failed: " << error << '\n';
        return 1;
    }
    std::cout << "features=" << matcher.featureCount() << " levels=" << matcher.levelCount()
              << " poses=" << stats.coarsePoses << " peaks=" << stats.coarsePeaks
              << " ms=" << stats.elapsedMs << " response=" << stats.responseMs
              << " coarse=" << stats.coarseScanMs << " refine=" << stats.refineMs
              << " matches=" << matches.size() << '\n';
    for (const auto& match : matches) {
        std::cout << "x=" << match.center.x << " y=" << match.center.y
                  << " angle=" << match.angleDeg << " scale=" << match.scale
                  << " score=" << match.score << '\n';
    }
    // rings_06 中齿形环的中心约为 (338, 114)。
    bool realImageFound = false;
    for (const auto& match : matches) {
        if (std::hypot(match.center.x - 338.0f, match.center.y - 114.0f) < 18.0f &&
            match.score >= options.minScore) realImageFound = true;
    }
    if (!realImageFound) {
        std::cerr << "Did not localize the toothed ring in rings_06.png\n";
        return 1;
    }

    // 并行搜索按行分工；单线程和多线程应给出相同顺序与分数。
    const int originalThreads = cv::getNumThreads();
    cv::setNumThreads(1);
    const auto serialMatches = matcher.find(sceneImage, options, error);
    cv::setNumThreads(originalThreads);
    if (!error.empty() || serialMatches.size() != matches.size()) {
        std::cerr << "Serial and parallel searches differ in result count\n";
        return 1;
    }
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& a = matches[i];
        const auto& b = serialMatches[i];
        if (cv::norm(a.center - b.center) > 1e-5f ||
            std::abs(a.angleDeg - b.angleDeg) > 1e-5f ||
            std::abs(a.scale - b.scale) > 1e-5f ||
            std::abs(a.score - b.score) > 1e-6f) {
            std::cerr << "Serial and parallel searches differ in pose or score\n";
            return 1;
        }
    }

    // 开口挡圈不存在齿形环的旋转对称歧义，适合检验角度和尺度估计。
    const cv::Rect clipRoi(95, 145, 205, 185);
    patmax::ShapeMatcher clipMatcher;
    if (!clipMatcher.create(modelImage, clipRoi, options, error)) {
        std::cerr << "Clip model creation failed: " << error << '\n';
        return 1;
    }
    patmax::Options fastClipOptions = options;
    fastClipOptions.maxLevels = 5;
    fastClipOptions.fastRefine = true;
    patmax::ShapeMatcher fastClipMatcher;
    if (!fastClipMatcher.create(modelImage, clipRoi, fastClipOptions, error)) {
        std::cerr << "Fast clip model creation failed: " << error << '\n';
        return 1;
    }
    const cv::Mat patch = modelImage(clipRoi);
    const cv::Point2f sourceCenter((patch.cols - 1) * 0.5f, (patch.rows - 1) * 0.5f);
    for (const auto& sample : {std::tuple<float, float, cv::Point2f>{37.5f, 0.9f, {280, 250}},
                               std::tuple<float, float, cv::Point2f>{-67.5f, 1.1f, {445, 280}}}) {
        const float angle = std::get<0>(sample);
        const float scale = std::get<1>(sample);
        const cv::Point2f center = std::get<2>(sample);
        cv::Mat synthetic;
        cv::Mat matrix = cv::getRotationMatrix2D(sourceCenter, -angle, scale);
        matrix.at<double>(0, 2) += center.x - sourceCenter.x;
        matrix.at<double>(1, 2) += center.y - sourceCenter.y;
        cv::warpAffine(patch, synthetic, matrix, modelImage.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(255));
        const auto found = clipMatcher.find(synthetic, options, error);
        bool correct = false;
        for (const auto& match : found) {
            const float angleError = std::abs(std::remainder(match.angleDeg - angle, 360.0f));
            if (cv::norm(match.center - center) < 8.0f && angleError < 10.0f &&
                std::abs(match.scale - scale) < 0.09f) correct = true;
        }
        if (!correct) {
            std::cerr << "Failed synthetic pose: angle=" << angle << " scale=" << scale
                      << " center=" << center << '\n';
            for (const auto& match : found)
                std::cerr << "  got " << match.center << " / " << match.angleDeg << " / "
                          << match.scale << " / " << match.score << '\n';
            return 1;
        }
        const auto fastFound = fastClipMatcher.find(synthetic, fastClipOptions, error);
        bool fastCorrect = false;
        for (const auto& match : fastFound) {
            const float angleError = std::abs(std::remainder(match.angleDeg - angle, 360.0f));
            if (cv::norm(match.center - center) < 8.0f && angleError < 10.0f &&
                std::abs(match.scale - scale) < 0.09f) fastCorrect = true;
        }
        if (!error.empty() || !fastCorrect) {
            std::cerr << "Fast mode failed synthetic pose: angle=" << angle
                      << " scale=" << scale << " error=" << error << '\n';
            return 1;
        }
    }
    for (int i = 2; i <= 8; ++i) {
        const std::string name = "rings_0" + std::to_string(i) + ".png";
        const cv::Mat image = cv::imread(base + "/" + name, cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        patmax::SearchStats frameStats;
        const auto found = matcher.find(image, options, error, &frameStats);
        if (!error.empty()) {
            std::cerr << name << ": " << error << '\n';
            return 1;
        }
        if (found.empty()) {
            std::cerr << name << ": no toothed ring found\n";
            return 1;
        }
        std::cout << name << " matches=" << found.size() << " ms=" << frameStats.elapsedMs;
        if (!found.empty())
            std::cout << " best=(" << found.front().center.x << "," << found.front().center.y
                      << ") score=" << found.front().score;
        std::cout << '\n';
    }

    // 快速模式使用更深金字塔与局部细化，验证其在多张实际图上仍能定位。
    patmax::Options fastOptions = options;
    fastOptions.maxLevels = 5;
    fastOptions.fastRefine = true;
    patmax::ShapeMatcher fastMatcher;
    if (!fastMatcher.create(modelImage, cv::Rect(330, 110, 210, 210), fastOptions, error)) {
        std::cerr << "Fast model creation failed: " << error << '\n';
        return 1;
    }
    for (int i = 2; i <= 8; ++i) {
        const std::string name = "rings_0" + std::to_string(i) + ".png";
        const cv::Mat image = cv::imread(base + "/" + name, cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        patmax::SearchStats fastStats;
        const auto found = fastMatcher.find(image, fastOptions, error, &fastStats);
        if (!error.empty() || found.empty()) {
            patmax::Options deepFull = fastOptions;
            deepFull.fastRefine = false;
            const auto fullFound = fastMatcher.find(image, deepFull, error);
            std::cerr << name << ": fast mode did not find the toothed ring; built="
                      << fastMatcher.levelCount() << " peaks=" << fastStats.coarsePeaks
                      << " deepFull=" << fullFound.size() << '\n';
            return 1;
        }
    }
    return 0;
}
