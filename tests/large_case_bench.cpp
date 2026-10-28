#include "ShapeMatcher.h"

#include <QCoreApplication>
#include <QImageReader>
#include <QStringList>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>

namespace {

/**
 * @brief 用 Qt 读取含中文路径的图片，并复制为 OpenCV 单通道图像。
 * @param path 输入的图片路径。
 * @return 8 位灰度图；读取失败时为空。
 */
cv::Mat readGray(const QString& path) {
    QImageReader reader(path);
    const QImage image = reader.read().convertToFormat(QImage::Format_Grayscale8);
    if (image.isNull()) return {};
    return cv::Mat(image.height(), image.width(), CV_8UC1,
                   const_cast<uchar*>(image.constBits()), image.bytesPerLine()).clone();
}

/**
 * @brief 按核心算法相同的规则，把梯度方向转换为量化索引。
 * @param angle 输入的梯度方向，单位为弧度。
 * @param options 输入的方向量化数与极性选项。
 * @return [0, orientationBins) 范围内的方向索引。
 */
int orientationBin(float angle, const patmax::Options& options) {
    constexpr float pi = 3.14159265358979323846f;
    const float period = options.polarity == patmax::Polarity::Signed ? 2.0f * pi : pi;
    angle = std::fmod(angle, period);
    if (angle < 0.0f) angle += period;
    return static_cast<int>(std::floor(angle * options.orientationBins / period + 0.5f)) %
           options.orientationBins;
}

/**
 * @brief 统计最优匹配的模板点在待测图中得到的方向响应，解释最终分数。
 * @param templ 输入的模板整图。
 * @param scene 输入的待测整图。
 * @param roi 输入的模板 ROI，坐标属于模板整图。
 * @param matcher 输入的已建模型，提供实际参与评分的模板点位置。
 * @param match 输入的待解释匹配姿态。
 * @param options 输入的边缘及方向量化参数。
 * @return 无；输出各响应分值的特征点数量与复算平均分。
 */
void explainScore(const cv::Mat& templ, const cv::Mat& scene, cv::Rect roi,
                  const patmax::ShapeMatcher& matcher, const patmax::Match& match,
                  const patmax::Options& options) {
    cv::Mat templateBlurred, templateDx, templateDy;
    cv::GaussianBlur(templ(roi), templateBlurred, cv::Size(3, 3), 0.8);
    cv::Sobel(templateBlurred, templateDx, CV_32F, 1, 0, 3);
    cv::Sobel(templateBlurred, templateDy, CV_32F, 0, 1, 3);

    constexpr float pi = 3.14159265358979323846f;
    const float radians = match.angleDeg * pi / 180.0f;
    const float c = std::cos(radians) * match.scale;
    const float s = std::sin(radians) * match.scale;
    const cv::Point2f modelCenter((roi.width - 1) * 0.5f, (roi.height - 1) * 0.5f);
    std::vector<cv::Point> scenePoints;
    scenePoints.reserve(matcher.displayPoints().size());
    int left = scene.cols, top = scene.rows, right = 0, bottom = 0;
    for (const auto& modelPoint : matcher.displayPoints()) {
        const cv::Point2f relative = modelPoint - modelCenter;
        const cv::Point point(cvRound(match.center.x) + cvRound(c * relative.x - s * relative.y),
                              cvRound(match.center.y) + cvRound(s * relative.x + c * relative.y));
        scenePoints.push_back(point);
        left = std::min(left, point.x);
        top = std::min(top, point.y);
        right = std::max(right, point.x);
        bottom = std::max(bottom, point.y);
    }
    // 额外边界使局部高斯滤波和 Canny 与全图计算保持一致。
    const cv::Rect region = cv::Rect(left - 16, top - 16,
                                     right - left + 33, bottom - top + 33) &
                            cv::Rect(0, 0, scene.cols, scene.rows);
    cv::Mat sceneBlurred, sceneDx, sceneDy, edges;
    cv::GaussianBlur(scene(region), sceneBlurred, cv::Size(3, 3), 0.8);
    cv::Sobel(sceneBlurred, sceneDx, CV_16S, 1, 0, 3);
    cv::Sobel(sceneBlurred, sceneDy, CV_16S, 0, 1, 3);
    cv::Canny(sceneDx, sceneDy, edges, options.lowThreshold, options.highThreshold, false);

    const float period = options.polarity == patmax::Polarity::Signed ? 2.0f * pi : pi;
    const int minSquared = options.minContrast * options.minContrast;
    std::map<int, int, std::greater<int>> histogram;
    int noEdge = 0;
    std::array<int, 4> nearestAny{};
    std::array<int, 4> nearestAligned{};
    std::array<int, 4> pointsByHeight{};
    std::array<int, 4> missingByHeight{};
    int sum = 0;
    const auto& modelPoints = matcher.displayPoints();
    for (size_t i = 0; i < modelPoints.size(); ++i) {
        const int mx = cvRound(modelPoints[i].x);
        const int my = cvRound(modelPoints[i].y);
        const int heightPart = std::min(3, 4 * my / roi.height);
        ++pointsByHeight[heightPart];
        const float modelAngle = std::atan2(templateDy.at<float>(my, mx),
                                            templateDx.at<float>(my, mx)) + radians;
        const int modelBin = orientationBin(modelAngle, options);
        const cv::Point center = scenePoints[i] - region.tl();
        int best = 0;
        bool hasEdge = false;
        for (int dy = -1; dy <= 1; ++dy) {
            const int y = center.y + dy;
            if (y < 0 || y >= region.height) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = center.x + dx;
                if (x < 0 || x >= region.width || !edges.at<uchar>(y, x)) continue;
                const int gx = sceneDx.at<short>(y, x);
                const int gy = sceneDy.at<short>(y, x);
                if (gx * gx + gy * gy < minSquared) continue;
                hasEdge = true;
                const int sceneBin = orientationBin(std::atan2(float(gy), float(gx)), options);
                int distance = std::abs(modelBin - sceneBin);
                distance = std::min(distance, options.orientationBins - distance);
                const float cosine = std::max(0.0f,
                    std::cos(period * distance / options.orientationBins));
                best = std::max(best, int(cv::saturate_cast<uchar>(255.0f * cosine * cosine)));
            }
        }
        if (!hasEdge) {
            ++noEdge;
            ++missingByHeight[heightPart];
            int anyDistance = 9;
            int alignedDistance = 9;
            for (int dy = -8; dy <= 8; ++dy) {
                const int y = center.y + dy;
                if (y < 0 || y >= region.height) continue;
                for (int dx = -8; dx <= 8; ++dx) {
                    const int x = center.x + dx;
                    if (x < 0 || x >= region.width || !edges.at<uchar>(y, x)) continue;
                    const int gx = sceneDx.at<short>(y, x);
                    const int gy = sceneDy.at<short>(y, x);
                    if (gx * gx + gy * gy < minSquared) continue;
                    const int distance = std::max(std::abs(dx), std::abs(dy));
                    anyDistance = std::min(anyDistance, distance);
                    const int sceneBin = orientationBin(std::atan2(float(gy), float(gx)), options);
                    int binDistance = std::abs(modelBin - sceneBin);
                    binDistance = std::min(binDistance, options.orientationBins - binDistance);
                    if (binDistance <= 1) alignedDistance = std::min(alignedDistance, distance);
                }
            }
            const auto category = [](int distance) {
                return distance <= 2 ? 0 : distance <= 4 ? 1 : distance <= 8 ? 2 : 3;
            };
            ++nearestAny[category(anyDistance)];
            ++nearestAligned[category(alignedDistance)];
        }
        ++histogram[best];
        sum += best;
    }
    std::cout << "  explain points=" << modelPoints.size() << " noEdge=" << noEdge
              << " recalculated=" << std::setprecision(6)
              << static_cast<double>(sum) / (255.0 * modelPoints.size());
    for (const auto& [response, count] : histogram)
        std::cout << " response" << response << "=" << count;
    std::cout << "\n  missing-edge nearest valid [2,4,8,>8]=";
    for (const int count : nearestAny) std::cout << count << ',';
    std::cout << " nearest aligned [2,4,8,>8]=";
    for (const int count : nearestAligned) std::cout << count << ',';
    std::cout << "\n  ROI height quarters missing/total=";
    for (int i = 0; i < 4; ++i)
        std::cout << missingByHeight[i] << '/' << pointsByHeight[i] << ',';
    std::cout << std::setprecision(2) << '\n';
}

} // namespace

/**
 * @brief 用一对实际图片比较不同金字塔层数的定位结果和阶段耗时。
 * @param argc 命令行参数个数。
 * @param argv 依次传入模板路径、待测路径和 ROI 的 x、y、宽、高。
 * @return 0 为正常结束，非零为参数、读取或建模错误。
 */
int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    const QStringList args = app.arguments();
    if (args.size() < 7 || args.size() > 9) {
        std::cerr << "Usage: patmax_bench <template> <scene> <x> <y> <w> <h> [--fast] [--explain]\n";
        return 2;
    }
    const cv::Mat templ = readGray(args[1]);
    const cv::Mat scene = readGray(args[2]);
    if (templ.empty() || scene.empty()) {
        std::cerr << "Cannot read input images\n";
        return 2;
    }
    const cv::Rect roi(args[3].toInt(), args[4].toInt(), args[5].toInt(), args[6].toInt());
    const bool fast = args.contains("--fast");
    const bool explain = args.contains("--explain");
    std::cout << "image=" << scene.cols << "x" << scene.rows
              << " roi=" << roi.x << "," << roi.y << "," << roi.width << "," << roi.height
              << '\n';
    for (int maxLevels = 3; maxLevels <= 6; ++maxLevels) {
        patmax::Options options;
        options.maxLevels = maxLevels;
        options.scaleMin = options.scaleMax = 1.0f;
        options.fastRefine = fast;
        patmax::ShapeMatcher matcher;
        std::string error;
        if (!matcher.create(templ, roi, options, error)) {
            std::cerr << "maxLevels=" << maxLevels << " create failed: " << error << '\n';
            return 1;
        }
        patmax::SearchStats stats;
        const auto matches = matcher.find(scene, options, error, &stats);
        if (!error.empty()) {
            std::cerr << "maxLevels=" << maxLevels << " find failed: " << error << '\n';
            return 1;
        }
        std::cout << std::fixed << std::setprecision(2)
                  << "maxLevels=" << maxLevels << " fast=" << fast
                  << " built=" << matcher.levelCount()
                  << " coarsePoints=" << matcher.featureCount(matcher.levelCount() - 1)
                  << " poses=" << stats.coarsePoses << " peaks=" << stats.coarsePeaks
                  << " matches=" << matches.size() << " total=" << stats.elapsedMs
                  << " prep=" << stats.responseMs << " scan=" << stats.coarseScanMs
                  << " refine=" << stats.refineMs;
        if (!matches.empty()) {
            const auto& best = matches.front();
            std::cout << " best=(" << best.center.x << "," << best.center.y << ")"
                      << " angle=" << best.angleDeg << " score="
                      << std::setprecision(6) << best.score << std::setprecision(2);
        }
        std::cout << '\n';
        for (const auto& level : stats.levels) {
            std::cout << "  level=" << level.index << " candidates=" << level.candidates
                      << " aboveFloor=" << level.aboveCoarseFloor
                      << " aboveMin=" << level.aboveMinScore
                      << " prep=" << level.responseMs << " refine=" << level.refineMs << '\n';
        }
        if (explain && maxLevels == 5 && !matches.empty())
            explainScore(templ, scene, roi, matcher, matches.front(), options);
    }
    return 0;
}
