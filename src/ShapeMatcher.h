#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace patmax {

// Signed 保留梯度方向；IgnoreLocal 将相反方向视为等价，适用于目标明暗反转。
enum class Polarity { Signed, IgnoreLocal };

struct Options {
    // Canny 阈值、Sobel 最小梯度，以及每层最多保留的轮廓点数量。
    int lowThreshold = 25;
    int highThreshold = 70;
    int minContrast = 16;
    int maxFeatures = 180;
    int orientationBins = 16;
    int maxLevels = 3;
    // 角度单位为度；尺度是相对模板原尺寸的等比例倍数。
    float angleMinDeg = -180.0f;
    float angleMaxDeg = 180.0f;
    float angleStepDeg = 15.0f;
    float scaleMin = 0.8f;
    float scaleMax = 1.2f;
    float scaleStep = 0.1f;
    float minScore = 0.58f;
    float maxOverlap = 0.35f;
    int maxMatches = 20;
    Polarity polarity = Polarity::Signed;
    // 快速细化只保留较可信的候选，并在低层局部计算响应；弱目标可能漏检。
    bool fastRefine = false;
};

struct Match {
    // 中心坐标以待测图像像素为单位；图像 Y 轴向下，正角度在屏幕上为顺时针。
    cv::Point2f center;
    float angleDeg = 0.0f;
    float scale = 1.0f;
    float score = 0.0f;
};

struct SearchStats {
    struct Level {
        int index = 0;
        int candidates = 0;
        int aboveCoarseFloor = 0;
        int aboveMinScore = 0;
        double responseMs = 0.0;
        double refineMs = 0.0;
    };
    int coarseLevel = 0;
    int coarsePoses = 0;
    int coarsePeaks = 0;
    // 以下耗时单位为毫秒；局部模式下的局部图像预处理计入 refineMs。
    double responseMs = 0.0;
    double coarseScanMs = 0.0;
    double refineMs = 0.0;
    double elapsedMs = 0.0;
    // 各细化层的候选分数与耗时，用于定位大图搜索瓶颈。
    std::vector<Level> levels;
};

class ShapeMatcher {
public:
    /**
     * @brief 从模板 ROI 建立多层边缘梯度模型；ROI 中心是后续旋转中心。
     * @param gray 输入的 8 位单通道模板图像。
     * @param roi 输入的模板矩形，使用整张模板图的像素坐标。
     * @param options 输入的边缘提取、采样和金字塔参数。
     * @param error 输出的错误信息；成功时清空。
     * @param mask 可选输入掩膜，大小可等于整图或 ROI；非零位置参与建模。
     * @return 建模成功为 true；参数或有效边缘不足时为 false。
     */
    bool create(const cv::Mat& gray, cv::Rect roi, const Options& options,
                std::string& error, const cv::Mat& mask = {});
    /**
     * @brief 在待测图中搜索模板的平移、平面旋转和等比例缩放姿态。
     * @param gray 输入的 8 位单通道待测图像。
     * @param options 输入的搜索范围、分数阈值和重复结果抑制参数。
     * @param error 输出的错误信息；成功时清空。
     * @param stats 可选输出，记录耗时和粗层候选数量。
     * @return 按分数从高到低排列的匹配结果；无匹配或失败时为空。
     */
    std::vector<Match> find(const cv::Mat& gray, const Options& options,
                            std::string& error, SearchStats* stats = nullptr) const;
    /// @return 当前是否拥有可搜索的模型。
    bool ready() const { return !levels_.empty(); }
    /// @return 建模时使用的整图像素 ROI。
    cv::Rect roi() const { return roi_; }
    /// @return 第 0 层特征点在 ROI 内的像素坐标，用于界面绘制。
    const std::vector<cv::Point2f>& displayPoints() const { return displayPoints_; }
    /// @return 成功建立的金字塔层数。
    int levelCount() const { return static_cast<int>(levels_.size()); }
    /// @param level 金字塔层号，0 为原图层。
    /// @return 指定层的特征点数量；层号无效时为 0。
    int featureCount(int level = 0) const;

private:
    struct Feature {
        // 特征点坐标相对于当前金字塔层的 ROI 中心。
        cv::Point2f point;
        float angleRad = 0.0f;
        float strength = 0.0f;
    };
    struct LevelModel {
        cv::Size size;
        std::vector<Feature> features;
    };
    cv::Rect roi_;
    std::vector<LevelModel> levels_;
    std::vector<cv::Point2f> displayPoints_;
};

} // namespace patmax
