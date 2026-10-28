#pragma once

#include "ShapeMatcher.h"

#include <QImage>
#include <QRect>
#include <QWidget>

#include <functional>
#include <utility>

class ImageView : public QWidget {
public:
    /// @param parent Qt 父控件；Qt Designer 会通过此构造函数创建图像视图。
    explicit ImageView(QWidget* parent = nullptr);
    /// @param selectable 是否允许鼠标拖选模板 ROI。
    void setSelectable(bool selectable);
    /// @param image 要显示的原图；同时清除旧 ROI 和匹配标记。
    void setImage(const QImage& image);
    /// @param selection 原图像素坐标中的 ROI；更新后通知选择回调。
    void setSelection(const QRect& selection);
    /// @return 当前 ROI，单位为原图像素。
    QRect selection() const { return selection_; }
    /// @param points 模型特征点在整张模板图中的像素坐标。
    void setTemplatePoints(const std::vector<cv::Point2f>& points);
    /**
     * @brief 在待测图上显示匹配位置、旋转框和对应的边缘特征点。
     * @param matches 匹配结果，中心坐标使用待测图像素。
     * @param modelSize 模板 ROI 原始宽高，单位为像素。
     * @param modelPoints 特征点在模板 ROI 内的像素坐标。
     */
    void setMatches(const std::vector<patmax::Match>& matches, const QSize& modelSize,
                    const std::vector<cv::Point2f>& modelPoints);
    /// @param callback ROI 变化时调用的函数；输入参数是原图像素 ROI。
    void setSelectionCallback(std::function<void(const QRect&)> callback) {
        onSelectionChanged_ = std::move(callback);
    }

protected:
    /// @brief 绘制等比例缩放的图像，以及 ROI、特征点和匹配标记。
    void paintEvent(QPaintEvent*) override;
    /// @brief 鼠标左键在图像范围内按下时开始拖选 ROI。
    void mousePressEvent(QMouseEvent*) override;
    /// @brief 拖动时更新预览 ROI。
    void mouseMoveEvent(QMouseEvent*) override;
    /// @brief 松开鼠标时确定 ROI 并通知回调。
    void mouseReleaseEvent(QMouseEvent*) override;

private:
    /// @return 原图等比例适配后的控件区域，使用控件坐标。
    QRectF imageRect() const;
    /// @param widgetPoint 控件坐标中的鼠标位置。
    /// @return 对应的原图像素坐标，并限制在图像范围内。
    QPoint imagePoint(const QPoint& widgetPoint) const;
    bool selectable_ = false;
    bool dragging_ = false;
    QPoint dragStart_;
    QImage image_;
    QRect selection_;
    std::vector<cv::Point2f> templatePoints_;
    std::vector<patmax::Match> matches_;
    QSize modelSize_;
    std::vector<cv::Point2f> modelPoints_;
    std::function<void(const QRect&)> onSelectionChanged_;
};
