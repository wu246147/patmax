#include "ShapeMatcher.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace patmax {
namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Candidate {
    cv::Point point;
    float angle = 0.0f;
    float strength = 0.0f;
};

struct PreparedPoint {
    int dx = 0;
    int dy = 0;
    int bin = 0;
};

struct PreparedPose {
    float angle = 0.0f;
    float scale = 1.0f;
    std::vector<PreparedPoint> points;
};

struct ScanPoint {
    const uchar* response = nullptr;
    int offset = 0;
};

struct PoseScan {
    std::vector<ScanPoint> points;
    int x0 = 0;
    int x1 = -1;
    int y0 = 0;
    int y1 = -1;
    int required = 0;
};

struct OrientationTable {
    // 低 16 位和高 16 位分别查表，避免为 32 位掩码创建 2^32 项表。
    std::vector<uchar> low;
    std::vector<uchar> high;
};

constexpr int kMaskLutSize = 1 << 16;

/**
 * @brief 将梯度弧度角量化为方向索引。
 * @param angle 输入梯度方向，单位为弧度。
 * @param bins 方向量化区间数。
 * @param polarity 是否区分相反的梯度方向。
 * @return 范围为 [0, bins) 的方向索引。
 */
int orientationBin(float angle, int bins, Polarity polarity) {
    // 忽略局部极性时，相差 π 的两个梯度方向落入同一量化区间。
    const float period = polarity == Polarity::Signed ? 2.0f * kPi : kPi;
    angle = std::fmod(angle, period);
    if (angle < 0.0f) angle += period;
    return static_cast<int>(std::floor(angle * bins / period + 0.5f)) % bins;
}

/**
 * @brief 预计算模板方向与图像方向的相似度。
 * @param options 输入的方向量化数和极性设置。
 * @return 第一维为模板方向、第二维为图像方向的 0–255 分数表。
 */
std::vector<std::vector<uchar>> directionScores(const Options& options) {
    const int bins = options.orientationBins;
    const float period = options.polarity == Polarity::Signed ? 2.0f * kPi : kPi;
    std::vector<std::vector<uchar>> lut(bins, std::vector<uchar>(bins));
    for (int modelBin = 0; modelBin < bins; ++modelBin) {
        for (int imageBin = 0; imageBin < bins; ++imageBin) {
            int d = std::abs(modelBin - imageBin);
            d = std::min(d, bins - d);
            const float cosine = std::max(0.0f, std::cos(period * d / bins));
            lut[modelBin][imageBin] = cv::saturate_cast<uchar>(255.0f * cosine * cosine);
        }
    }
    return lut;
}

/**
 * @brief 将 3×3 邻域内的方向集合映射为最佳方向分数。
 * @param options 输入的方向量化数和极性设置。
 * @return 每个模板方向对应两张 16 位掩码查找表。
 */
OrientationTable makeOrientationTable(const Options& options) {
    const auto scores = directionScores(options);
    const int bins = options.orientationBins;
    OrientationTable table;
    table.low.resize(static_cast<size_t>(bins) * kMaskLutSize);
    if (bins > 16) table.high.resize(static_cast<size_t>(bins) * kMaskLutSize);
    std::vector<uchar> leastBit(kMaskLutSize, 0);
    for (int mask = 2; mask < kMaskLutSize; ++mask)
        if (!(mask & 1)) leastBit[mask] = static_cast<uchar>(leastBit[mask >> 1] + 1);
    for (int modelBin = 0; modelBin < bins; ++modelBin) {
        uchar* low = table.low.data() + static_cast<size_t>(modelBin) * kMaskLutSize;
        uchar* high = table.high.empty() ? nullptr :
            table.high.data() + static_cast<size_t>(modelBin) * kMaskLutSize;
        for (int mask = 1; mask < kMaskLutSize; ++mask) {
            const int previous = mask & (mask - 1);
            const int imageBin = leastBit[mask];
            low[mask] = imageBin < bins ? std::max(low[previous], scores[modelBin][imageBin])
                                        : low[previous];
            if (high) {
                const int upperBin = imageBin + 16;
                high[mask] = upperBin < bins ?
                    std::max(high[previous], scores[modelBin][upperBin]) : high[previous];
            }
        }
    }
    return table;
}

struct GradientEdges {
    cv::Mat1b edges;
    cv::Mat1s dx;
    cv::Mat1s dy;
};

/**
 * @brief 计算 Canny 边缘及对应 Sobel 梯度，避免对同一图重复求导。
 * @param gray 输入的 8 位灰度图。
 * @param options 输入的 Canny 高低阈值。
 * @return 边缘掩码和 16 位梯度；供方向量化与最小梯度筛选使用。
 */
GradientEdges makeGradientEdges(const cv::Mat& gray, const Options& options) {
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.8);
    GradientEdges result;
    cv::Sobel(blurred, result.dx, CV_16S, 1, 0, 3);
    cv::Sobel(blurred, result.dy, CV_16S, 0, 1, 3);
    cv::Canny(result.dx, result.dy, result.edges,
              options.lowThreshold, options.highThreshold, false);
    return result;
}

/**
 * @brief 为每个模板方向建立待测图响应图，供后续快速查表评分。
 * @param gray 输入的 8 位灰度待测图。
 * @param options 边缘阈值、梯度阈值、方向量化数和极性设置。
 * @return 每个方向一张 8 位响应图；0 表示无可用边缘，255 表示方向完全一致。
 */
std::vector<cv::Mat1b> makeResponses(const cv::Mat& gray, const Options& options) {
    // 仅边缘像素产生方向响应；非边缘位置始终为零。
    const GradientEdges gradient = makeGradientEdges(gray, options);
    const int minMagnitudeSquared = options.minContrast * options.minContrast;

    const int bins = options.orientationBins;
    // 用正余弦的平方衡量方向一致性，使相邻方向区间的分数平滑过渡。
    const auto lut = directionScores(options);

    std::vector<cv::Mat1b> responses;
    responses.reserve(bins);
    for (int b = 0; b < bins; ++b) responses.emplace_back(gray.size(), uchar(0));

    for (int y = 0; y < gray.rows; ++y) {
        const uchar* ep = gradient.edges.ptr<uchar>(y);
        const short* gx = gradient.dx.ptr<short>(y);
        const short* gy = gradient.dy.ptr<short>(y);
        std::vector<uchar*> rows(bins);
        for (int b = 0; b < bins; ++b) rows[b] = responses[b].ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            if (!ep[x] || gx[x] * gx[x] + gy[x] * gy[x] < minMagnitudeSquared) continue;
            const int imageBin = orientationBin(std::atan2(float(gy[x]), float(gx[x])),
                                                bins, options.polarity);
            for (int b = 0; b < bins; ++b) rows[b][x] = lut[b][imageBin];
        }
    }

    // 允许 1 像素位置误差，减轻 Canny 非极大值抑制和金字塔取整的影响。
    for (auto& response : responses) {
        cv::Mat1b dilated;
        cv::dilate(response, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        response = std::move(dilated);
    }
    return responses;
}

/**
 * @brief 用单张 32 位掩码图保存每个像素 3×3 邻域出现的梯度方向。
 * @param gray 输入的 8 位灰度图。
 * @param options 输入的边缘阈值、最小梯度和方向量化参数。
 * @return 位 i 表示邻域中存在第 i 个方向的有效边缘。
 */
cv::Mat1i makePackedResponses(const cv::Mat& gray, const Options& options) {
    const GradientEdges gradient = makeGradientEdges(gray, options);
    const int minMagnitudeSquared = options.minContrast * options.minContrast;

    cv::Mat1i edgeBits(gray.size(), 0);
    for (int y = 0; y < gray.rows; ++y) {
        const uchar* ep = gradient.edges.ptr<uchar>(y);
        const short* gx = gradient.dx.ptr<short>(y);
        const short* gy = gradient.dy.ptr<short>(y);
        int* bits = edgeBits.ptr<int>(y);
        for (int x = 0; x < gray.cols; ++x) {
            if (!ep[x] || gx[x] * gx[x] + gy[x] * gy[x] < minMagnitudeSquared) continue;
            const int bin = orientationBin(std::atan2(float(gy[x]), float(gx[x])),
                                           options.orientationBins, options.polarity);
            bits[x] = static_cast<int>(uint32_t{1} << bin);
        }
    }

    // 水平和垂直各合并一次，等价于对每张方向响应图作 3×3 最大值膨胀。
    cv::Mat1i horizontal(gray.size(), 0), packed(gray.size(), 0);
    for (int y = 0; y < gray.rows; ++y) {
        const int* source = edgeBits.ptr<int>(y);
        int* target = horizontal.ptr<int>(y);
        for (int x = 0; x < gray.cols; ++x) {
            target[x] = source[x] |
                        (x > 0 ? source[x - 1] : 0) |
                        (x + 1 < gray.cols ? source[x + 1] : 0);
        }
    }
    for (int y = 0; y < gray.rows; ++y) {
        const int* above = y > 0 ? horizontal.ptr<int>(y - 1) : nullptr;
        const int* middle = horizontal.ptr<int>(y);
        const int* below = y + 1 < gray.rows ? horizontal.ptr<int>(y + 1) : nullptr;
        int* target = packed.ptr<int>(y);
        for (int x = 0; x < gray.cols; ++x)
            target[x] = middle[x] | (above ? above[x] : 0) | (below ? below[x] : 0);
    }
    return packed;
}

/**
 * @brief 使用邻域方向位图计算细化阶段的梯度匹配分数。
 * @param points 当前旋转和尺度下的模板点。
 * @param packed 待测图中每个像素的邻域方向位图。
 * @param table 方向位图到最佳梯度分数的查找表。
 * @param x 候选中心横坐标。
 * @param y 候选中心纵坐标。
 * @return [0, 1] 内的平均梯度分数，越界特征贡献零分。
 */
float scoreAtPacked(const std::vector<PreparedPoint>& points, const cv::Mat1i& packed,
                    const OrientationTable& table, int x, int y) {
    int sum = 0;
    for (const auto& point : points) {
        const int px = x + point.dx;
        const int py = y + point.dy;
        if (static_cast<unsigned>(px) >= static_cast<unsigned>(packed.cols) ||
            static_cast<unsigned>(py) >= static_cast<unsigned>(packed.rows)) continue;
        const uint32_t mask = static_cast<uint32_t>(packed.ptr<int>(py)[px]);
        const size_t lutBase = static_cast<size_t>(point.bin) * kMaskLutSize;
        uchar response = table.low[lutBase + (mask & 0xffffu)];
        if (!table.high.empty())
            response = std::max(response, table.high[lutBase + (mask >> 16)]);
        sum += response;
    }
    return static_cast<float>(sum) / (255.0f * points.size());
}

/**
 * @brief 对保证全部模板点位于图内的候选中心快速评分。
 * @param points 预先绑定方向响应图和线性像素偏移的模板点。
 * @param centerIndex 候选中心在响应图中的线性像素索引。
 * @param required 最低分数换算后的整数响应总和，用于提前结束。
 * @return [0, 1] 内的平均分数；不可能达到最低分数时返回 0。
 */
float scoreInside(const std::vector<ScanPoint>& points, int centerIndex, int required) {
    int sum = 0;
    int remaining = static_cast<int>(points.size()) * 255;
    for (const auto& point : points) {
        sum += point.response[centerIndex + point.offset];
        remaining -= 255;
        if (sum + remaining < required) return 0.0f;
    }
    return static_cast<float>(sum) / (255.0f * points.size());
}

/**
 * @brief 计算两个匹配结果旋转外接矩形的交并比。
 * @param a 第一个候选姿态。
 * @param b 第二个候选姿态。
 * @param modelSize 模板 ROI 的原始宽高。
 * @return [0, 1] 内的交并比，用于抑制重复结果。
 */
float overlapRatio(const Match& a, const Match& b, cv::Size modelSize) {
    // 用旋转矩形的交并比去除相邻姿态下的重复目标。
    const cv::RotatedRect ra(a.center,
                             cv::Size2f(modelSize.width * a.scale, modelSize.height * a.scale),
                             a.angleDeg);
    const cv::RotatedRect rb(b.center,
                             cv::Size2f(modelSize.width * b.scale, modelSize.height * b.scale),
                             b.angleDeg);
    std::vector<cv::Point2f> polygon;
    const int kind = cv::rotatedRectangleIntersection(ra, rb, polygon);
    if (kind == cv::INTERSECT_NONE || polygon.size() < 3) return 0.0f;
    const float intersection = static_cast<float>(std::abs(cv::contourArea(polygon)));
    const float areaA = ra.size.area();
    const float areaB = rb.size.area();
    return intersection / std::max(1.0f, areaA + areaB - intersection);
}

} // namespace

int ShapeMatcher::featureCount(int level) const {
    if (level < 0 || level >= static_cast<int>(levels_.size())) return 0;
    return static_cast<int>(levels_[level].features.size());
}

bool ShapeMatcher::create(const cv::Mat& gray, cv::Rect roi, const Options& options,
                          std::string& error, const cv::Mat& mask) {
    error.clear();
    if (gray.type() != CV_8UC1 || gray.empty()) {
        error = "Template must be an 8-bit grayscale image.";
        return false;
    }
    if (roi.width < 30 || roi.height < 30 ||
        (roi & cv::Rect(0, 0, gray.cols, gray.rows)) != roi) {
        error = "Select a template rectangle of at least 30 x 30 pixels inside the image.";
        return false;
    }
    if (options.lowThreshold < 0 || options.highThreshold <= options.lowThreshold ||
        options.maxFeatures < 16 || options.maxLevels < 1 ||
        options.orientationBins < 4 || options.orientationBins > 32) {
        error = "Invalid model parameters.";
        return false;
    }
    cv::Mat currentMask;
    if (!mask.empty()) {
        if (mask.type() != CV_8UC1) {
            error = "Mask must be 8-bit grayscale.";
            return false;
        }
        if (mask.size() == gray.size()) currentMask = mask(roi).clone();
        else if (mask.size() == roi.size()) currentMask = mask.clone();
        else {
            error = "Mask must have the source-image or template-ROI size.";
            return false;
        }
    }

    std::vector<LevelModel> built;
    cv::Mat current = gray(roi).clone();
    for (int level = 0; level < options.maxLevels; ++level) {
        // 小模板的高层仍可用于粗定位；最终得分由低层完整轮廓复核。
        if (current.cols < 16 || current.rows < 16) break;
        cv::Mat blurred, edges, dx, dy, magnitude;
        cv::GaussianBlur(current, blurred, cv::Size(3, 3), 0.8);
        cv::Canny(blurred, edges, options.lowThreshold, options.highThreshold);
        cv::Sobel(blurred, dx, CV_32F, 1, 0, 3);
        cv::Sobel(blurred, dy, CV_32F, 0, 1, 3);
        cv::magnitude(dx, dy, magnitude);

        std::vector<Candidate> candidates;
        for (int y = 2; y < current.rows - 2; ++y) {
            const uchar* ep = edges.ptr<uchar>(y);
            const float* gx = dx.ptr<float>(y);
            const float* gy = dy.ptr<float>(y);
            const float* mag = magnitude.ptr<float>(y);
            const uchar* mp = currentMask.empty() ? nullptr : currentMask.ptr<uchar>(y);
            for (int x = 2; x < current.cols - 2; ++x) {
                if (!ep[x] || mag[x] < options.minContrast || (mp && !mp[x])) continue;
                candidates.push_back({{x, y}, std::atan2(gy[x], gx[x]), mag[x]});
            }
        }
        const int minFeatures = level == 0 ? 16 : 8;
        if (candidates.size() < static_cast<size_t>(minFeatures)) {
            if (level == 0) {
                error = "Too few stable edges. Enlarge the ROI or lower edge thresholds.";
                return false;
            }
            break;
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.strength > b.strength; });
        const int target = std::max(32, static_cast<int>(options.maxFeatures / std::pow(1.7, level)));
        LevelModel model;
        model.size = current.size();
        const cv::Point2f center((current.cols - 1) * 0.5f, (current.rows - 1) * 0.5f);
        // 按间距贪心选点，让特征覆盖整条轮廓，避免集中在局部高对比度边缘。
        const float spacing = std::max(2.0f, std::sqrt(static_cast<float>(current.total()) / target) * 0.58f);
        for (float allowed : {spacing, std::max(1.2f, spacing * 0.55f)}) {
            for (const auto& candidate : candidates) {
                if (static_cast<int>(model.features.size()) >= target) break;
                const cv::Point2f point = cv::Point2f(candidate.point) - center;
                bool close = false;
                for (const auto& selected : model.features) {
                    if (cv::norm(point - selected.point) < allowed) { close = true; break; }
                }
                if (!close) model.features.push_back({point, candidate.angle, candidate.strength});
            }
        }
        if (model.features.size() < static_cast<size_t>(minFeatures)) {
            if (level == 0) {
                error = "The selected contour has too few distributed edge points.";
                return false;
            }
            break;
        }
        built.push_back(std::move(model));
        if (level + 1 >= options.maxLevels) break;
        cv::Mat next;
        cv::pyrDown(current, next);
        current = std::move(next);
        if (!currentMask.empty()) {
            cv::Mat nextMask;
            cv::resize(currentMask, nextMask, current.size(), 0, 0, cv::INTER_NEAREST);
            currentMask = std::move(nextMask);
        }
    }
    roi_ = roi;
    levels_ = std::move(built);
    displayPoints_.clear();
    const cv::Point2f center((roi.width - 1) * 0.5f, (roi.height - 1) * 0.5f);
    for (const auto& feature : levels_[0].features) displayPoints_.push_back(feature.point + center);
    return true;
}

std::vector<Match> ShapeMatcher::find(const cv::Mat& gray, const Options& options,
                                      std::string& error, SearchStats* stats) const {
    error.clear();
    if (stats) *stats = {};
    if (!ready()) { error = "Create a model first."; return {}; }
    if (gray.empty() || gray.type() != CV_8UC1) {
        error = "Search image must be 8-bit grayscale.";
        return {};
    }
    if (options.angleStepDeg <= 0 || options.scaleStep <= 0 ||
        options.angleMinDeg > options.angleMaxDeg || options.scaleMin <= 0 ||
        options.scaleMin > options.scaleMax || options.minScore < 0 ||
        options.minScore > 1 || options.maxMatches < 1 ||
        options.orientationBins < 4 || options.orientationBins > 32 ||
        options.lowThreshold < 0 || options.highThreshold <= options.lowThreshold) {
        error = "Invalid search parameters.";
        return {};
    }
    const int angleCount = 1 + static_cast<int>((options.angleMaxDeg - options.angleMinDeg) / options.angleStepDeg);
    const int scaleCount = 1 + static_cast<int>((options.scaleMax - options.scaleMin) / options.scaleStep);
    if (static_cast<long long>(angleCount) * scaleCount > 2500) {
        error = "Search grid exceeds 2500 poses. Increase angle or scale step.";
        return {};
    }

    cv::TickMeter timer;
    timer.start();
    const double tickToMs = 1000.0 / cv::getTickFrequency();
    int64 responseTicks = 0;
    int64 fineResponseTicks = 0;
    const int64 pyramidStart = cv::getTickCount();
    const int coarse = std::min(static_cast<int>(levels_.size()), options.maxLevels) - 1;
    std::vector<cv::Mat> imagePyramid(coarse + 1);
    imagePyramid[0] = gray;
    for (int i = 1; i <= coarse; ++i) cv::pyrDown(imagePyramid[i - 1], imagePyramid[i]);

    auto prepare = [&](int level, float angleDeg, float scale) {
        // 每种旋转和尺度只变换一次 ROI 中心化特征；扫描中心时只加整数平移量。
        std::vector<PreparedPoint> prepared;
        prepared.reserve(levels_[level].features.size());
        const float a = angleDeg * kPi / 180.0f;
        const float c = std::cos(a) * scale;
        const float s = std::sin(a) * scale;
        for (const auto& feature : levels_[level].features) {
            prepared.push_back({cvRound(c * feature.point.x - s * feature.point.y),
                                cvRound(s * feature.point.x + c * feature.point.y),
                                orientationBin(feature.angleRad + a, options.orientationBins,
                                               options.polarity)});
        }
        return prepared;
    };

    auto responses = makeResponses(imagePyramid[coarse], options);
    responseTicks += cv::getTickCount() - pyramidStart;
    const int64 scanStart = cv::getTickCount();
    const cv::Size sceneSize = imagePyramid[coarse].size();
    const int stride = static_cast<int>(responses[0].step1());
    // 快速评分以线性索引读取响应图；所有方向图必须具有相同的连续存储步长。
    for (const auto& response : responses)
        CV_Assert(response.isContinuous() && response.step1() == stride);
    cv::Mat1f scoreMap(sceneSize, -1.0f);
    cv::Mat1i poseMap(sceneSize, -1);
    std::vector<std::pair<float, float>> poses;
    std::vector<PoseScan> scans;
    // 最粗层遍历全部角度和尺度；每个位置只保留最高分姿态，再提取局部峰值。
    const float candidateFloor = std::max(0.18f, options.minScore * 0.65f);
    for (int si = 0; si < scaleCount; ++si) {
        const float scale = std::min(options.scaleMax, options.scaleMin + si * options.scaleStep);
        for (int ai = 0; ai < angleCount; ++ai) {
            const float angle = std::min(options.angleMaxDeg,
                                         options.angleMinDeg + ai * options.angleStepDeg);
            if (ai == angleCount - 1 && ai > 0 &&
                std::abs(angle - (options.angleMinDeg + 360.0f)) < 0.001f) continue;
            poses.push_back({angle, scale});
            const auto points = prepare(coarse, angle, scale);
            int minDx = 0, maxDx = 0, minDy = 0, maxDy = 0;
            for (const auto& p : points) {
                minDx = std::min(minDx, p.dx); maxDx = std::max(maxDx, p.dx);
                minDy = std::min(minDy, p.dy); maxDy = std::max(maxDy, p.dy);
            }
            PoseScan scan;
            scan.x0 = std::max(0, -minDx);
            scan.y0 = std::max(0, -minDy);
            scan.x1 = std::min(sceneSize.width - 1, sceneSize.width - 1 - maxDx);
            scan.y1 = std::min(sceneSize.height - 1, sceneSize.height - 1 - maxDy);
            // 扫描范围已按旋转后的模板外接框收紧，所有特征点均在图内。
            // 因此可提前绑定响应图地址和偏移，内层循环只做连续索引。
            scan.points.reserve(points.size());
            for (const auto& point : points) {
                scan.points.push_back({responses[point.bin].data,
                                       point.dy * stride + point.dx});
            }
            scan.required = static_cast<int>(candidateFloor * points.size() * 255.0f);
            scans.push_back(std::move(scan));
        }
    }
    // 每个线程只写自己的行区间，保持与单线程相同的姿态遍历顺序和结果。
    const int stripes = std::min(sceneSize.height, std::max(1, cv::getNumThreads() * 2));
    cv::parallel_for_(cv::Range(0, sceneSize.height), [&](const cv::Range& rows) {
        for (int poseIndex = 0; poseIndex < static_cast<int>(scans.size()); ++poseIndex) {
            const auto& scan = scans[poseIndex];
            for (int y = std::max(rows.start, scan.y0);
                 y < std::min(rows.end, scan.y1 + 1); ++y) {
                float* scores = scoreMap.ptr<float>(y);
                int* ids = poseMap.ptr<int>(y);
                for (int x = scan.x0; x <= scan.x1; ++x) {
                    const float score = scoreInside(scan.points, y * stride + x, scan.required);
                    if (score > scores[x]) { scores[x] = score; ids[x] = poseIndex; }
                }
            }
        }
    }, stripes);

    const int64 scanEnd = cv::getTickCount();
    cv::Mat1f maxima;
    cv::dilate(scoreMap, maxima, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
    std::vector<Match> peaks;
    for (int y = 0; y < sceneSize.height; ++y) {
        const float* scores = scoreMap.ptr<float>(y);
        const float* maximums = maxima.ptr<float>(y);
        const int* ids = poseMap.ptr<int>(y);
        for (int x = 0; x < sceneSize.width; ++x) {
            if (ids[x] < 0 || scores[x] < candidateFloor || scores[x] + 1e-6f < maximums[x]) continue;
            const auto& pose = poses[ids[x]];
            peaks.push_back({cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                             pose.first, pose.second, scores[x]});
        }
    }
    std::sort(peaks.begin(), peaks.end(),
              [](const Match& a, const Match& b) { return a.score > b.score; });
    std::vector<Match> current;
    const int seedLimit = std::min(220, std::max(80, options.maxMatches * 15));
    for (const auto& peak : peaks) {
        const float separation = std::max(3.0f,
            std::min(levels_[coarse].size.width, levels_[coarse].size.height) * peak.scale * 0.12f);
        bool close = false;
        for (const auto& selected : current) {
            if (cv::norm(peak.center - selected.center) < separation) { close = true; break; }
        }
        if (!close) current.push_back(peak);
        if (static_cast<int>(current.size()) >= seedLimit) break;
    }
    if (stats) {
        stats->coarseLevel = coarse;
        stats->coarsePoses = static_cast<int>(poses.size());
        stats->coarsePeaks = static_cast<int>(current.size());
    }

    // 粗层扫描结束后释放多张方向图；细层改用单张方向位图评分。
    responses.clear();
    scans.clear();
    const int64 tableStart = cv::getTickCount();
    const OrientationTable table = coarse > 0 ? makeOrientationTable(options) : OrientationTable{};
    responseTicks += cv::getTickCount() - tableStart;
    const int64 refineStart = cv::getTickCount();
    for (int level = coarse - 1; level >= 0; --level) {
        // 上采样候选中心后，在位置、角度和尺度邻域细化，补回降采样误差。
        if (options.fastRefine && level <= 1 && current.size() > 1) {
            // 靠近原图的两层仅保留前一层达到粗筛分数的候选。
            // 这是可选的速度/召回率折中；至少保留最高分候选。
            const Match strongest = *std::max_element(current.begin(), current.end(),
                [](const Match& a, const Match& b) { return a.score < b.score; });
            const float cutoff = candidateFloor;
            current.erase(std::remove_if(current.begin(), current.end(),
                [cutoff](const Match& match) { return match.score < cutoff; }), current.end());
            if (current.empty()) current.push_back(strongest);
        }
        const bool useLocal = options.fastRefine && level <= 1 && current.size() <= 24;
        const int64 responseStart = cv::getTickCount();
        const cv::Mat1i packed = useLocal ? cv::Mat1i{} :
            makePackedResponses(imagePyramid[level], options);
        const int64 responseCost = cv::getTickCount() - responseStart;
        responseTicks += responseCost;
        fineResponseTicks += responseCost;
        const int64 refinementStart = cv::getTickCount();
        const int stage = coarse - level;
        const float angleDelta = options.angleStepDeg / static_cast<float>(1 << stage);
        const float scaleDelta = options.scaleStep / static_cast<float>(1 << stage);
        auto refineOne = [&](Match& match) {
            match.center *= 2.0f;
            Match best = match;
            best.score = -1.0f;
            std::vector<PreparedPose> variants;
            variants.reserve(9);
            int minDx = 0, maxDx = 0, minDy = 0, maxDy = 0;
            for (int ds = -1; ds <= 1; ++ds) {
                const float scale = match.scale + ds * scaleDelta;
                if (scale < options.scaleMin - 1e-5f || scale > options.scaleMax + 1e-5f) continue;
                for (int da = -1; da <= 1; ++da) {
                    const float angle = match.angleDeg + da * angleDelta;
                    if (angle < options.angleMinDeg - 1e-5f ||
                        angle > options.angleMaxDeg + 1e-5f) continue;
                    PreparedPose pose;
                    pose.angle = angle;
                    pose.scale = scale;
                    pose.points = prepare(level, angle, scale);
                    if (useLocal) {
                        for (const auto& point : pose.points) {
                            minDx = std::min(minDx, point.dx); maxDx = std::max(maxDx, point.dx);
                            minDy = std::min(minDy, point.dy); maxDy = std::max(maxDy, point.dy);
                        }
                    }
                    variants.push_back(std::move(pose));
                }
            }
            cv::Rect region(0, 0, imagePyramid[level].cols, imagePyramid[level].rows);
            cv::Mat1i localPacked;
            const cv::Mat1i* active = &packed;
            if (useLocal) {
                // 覆盖全部姿态和 ±3 像素细化范围，额外留边供 Canny 计算。
                const int padding = 16;
                const int cx = cvRound(match.center.x), cy = cvRound(match.center.y);
                const int left = std::max(0, cx + minDx - 3 - padding);
                const int top = std::max(0, cy + minDy - 3 - padding);
                const int right = std::min(imagePyramid[level].cols,
                                           cx + maxDx + 4 + padding);
                const int bottom = std::min(imagePyramid[level].rows,
                                            cy + maxDy + 4 + padding);
                if (right <= left || bottom <= top) { match.score = 0.0f; return; }
                region = cv::Rect(left, top, right - left, bottom - top);
                localPacked = makePackedResponses(imagePyramid[level](region), options);
                active = &localPacked;
            }
            for (const auto& pose : variants) {
                for (int dy = -3; dy <= 3; ++dy) {
                    const int y = cvRound(match.center.y) + dy;
                    if (y < 0 || y >= imagePyramid[level].rows) continue;
                    for (int dx = -3; dx <= 3; ++dx) {
                        const int x = cvRound(match.center.x) + dx;
                        if (x < 0 || x >= imagePyramid[level].cols) continue;
                        const float score = scoreAtPacked(pose.points, *active, table,
                                                           x - region.x, y - region.y);
                        if (score > best.score) {
                            best = {cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                                    pose.angle, pose.scale, score};
                        }
                    }
                }
            }
            match = best;
        };
        // 每个候选只修改自身的姿态，响应位图和查找表只读，可安全并行。
        if (current.size() >= 4) {
            const int stripes = std::min(static_cast<int>(current.size()),
                                         std::max(1, cv::getNumThreads() * 2));
            cv::parallel_for_(cv::Range(0, static_cast<int>(current.size())),
                              [&](const cv::Range& range) {
                for (int i = range.start; i < range.end; ++i) refineOne(current[i]);
            }, stripes);
        } else {
            for (auto& match : current) refineOne(match);
        }
        if (stats) {
            SearchStats::Level stageStats;
            stageStats.index = level;
            stageStats.candidates = static_cast<int>(current.size());
            stageStats.responseMs = responseCost * tickToMs;
            // 局部模式下的局部预处理计入细化耗时。
            stageStats.refineMs = (cv::getTickCount() - refinementStart) * tickToMs;
            for (const auto& match : current) {
                if (match.score >= candidateFloor) ++stageStats.aboveCoarseFloor;
                if (match.score >= options.minScore) ++stageStats.aboveMinScore;
            }
            stats->levels.push_back(stageStats);
        }
    }

    std::sort(current.begin(), current.end(),
              [](const Match& a, const Match& b) { return a.score > b.score; });
    std::vector<Match> result;
    // 按得分排序，再抑制覆盖同一目标的重复姿态。
    for (const auto& match : current) {
        if (match.score < options.minScore) continue;
        bool suppressed = false;
        for (const auto& selected : result) {
            if (overlapRatio(match, selected, roi_.size()) > options.maxOverlap) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) result.push_back(match);
        if (static_cast<int>(result.size()) >= options.maxMatches) break;
    }
    timer.stop();
    if (stats) {
        stats->responseMs = responseTicks * tickToMs;
        stats->coarseScanMs = (scanEnd - scanStart) * tickToMs;
        stats->refineMs = (cv::getTickCount() - refineStart - fineResponseTicks) * tickToMs;
        stats->elapsedMs = timer.getTimeMilli();
    }
    return result;
}

} // namespace patmax
