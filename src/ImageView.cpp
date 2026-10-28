#include "ImageView.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>

ImageView::ImageView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(280, 260);
}

void ImageView::setSelectable(bool selectable) {
    selectable_ = selectable;
    setMouseTracking(selectable);
    setCursor(selectable ? Qt::CrossCursor : Qt::ArrowCursor);
    update();
}

void ImageView::setImage(const QImage& image) {
    image_ = image;
    selection_ = {};
    templatePoints_.clear();
    matches_.clear();
    update();
}

void ImageView::setSelection(const QRect& selection) {
    selection_ = selection.normalized();
    update();
    if (onSelectionChanged_) onSelectionChanged_(selection_);
}

void ImageView::setTemplatePoints(const std::vector<cv::Point2f>& points) {
    templatePoints_ = points;
    update();
}

void ImageView::setMatches(const std::vector<patmax::Match>& matches,
                           const QSize& modelSize,
                           const std::vector<cv::Point2f>& modelPoints) {
    matches_ = matches;
    modelSize_ = modelSize;
    modelPoints_ = modelPoints;
    update();
}

QRectF ImageView::imageRect() const {
    if (image_.isNull()) return {};
    // 等比例居中显示完整图像，必要时留黑边。
    const double scale = std::min((width() - 16.0) / image_.width(),
                                  (height() - 16.0) / image_.height());
    const QSizeF size(image_.width() * scale, image_.height() * scale);
    return QRectF((width() - size.width()) * 0.5,
                  (height() - size.height()) * 0.5, size.width(), size.height());
}

QPoint ImageView::imagePoint(const QPoint& widgetPoint) const {
    const QRectF target = imageRect();
    if (target.isEmpty()) return {};
    // 逆向换算到原图坐标；ROI 始终使用原图像素单位。
    const int x = qBound(0, static_cast<int>((widgetPoint.x() - target.left()) *
                                             image_.width() / target.width()), image_.width() - 1);
    const int y = qBound(0, static_cast<int>((widgetPoint.y() - target.top()) *
                                             image_.height() / target.height()), image_.height() - 1);
    return {x, y};
}

void ImageView::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), QColor(34, 39, 46));
    if (image_.isNull()) {
        painter.setPen(QColor(215, 220, 225));
        painter.drawText(rect(), Qt::AlignCenter, selectable_ ?
                         tr("打开模板图后拖动鼠标框选目标") : tr("打开待测图像"));
        return;
    }
    const QRectF target = imageRect();
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.drawImage(target, image_);
    painter.save();
    painter.translate(target.topLeft());
    const double scale = target.width() / image_.width();
    painter.scale(scale, scale);
    // 以下标记统一在原图坐标系绘制。
    painter.setRenderHint(QPainter::Antialiasing);
    const qreal lineWidth = std::max<qreal>(1.0 / scale, 0.65);

    if (selectable_ && !selection_.isEmpty()) {
        painter.setPen(QPen(QColor(255, 200, 55), lineWidth * 1.5));
        painter.setBrush(QColor(255, 200, 55, 22));
        painter.drawRect(selection_);
    }
    if (!templatePoints_.empty()) {
        painter.setPen(QPen(QColor(0, 220, 255), lineWidth));
        for (const auto& point : templatePoints_)
            painter.drawPoint(QPointF(point.x, point.y));
    }

    const QPointF origin((modelSize_.width() - 1) * 0.5,
                         (modelSize_.height() - 1) * 0.5);
    // 模板点以 ROI 为局部坐标，围绕 ROI 中心旋转并缩放。
    for (size_t i = 0; i < matches_.size(); ++i) {
        const auto& match = matches_[i];
        const double rad = match.angleDeg * 3.14159265358979323846 / 180.0;
        const double c = std::cos(rad) * match.scale;
        const double s = std::sin(rad) * match.scale;
        auto transform = [&](QPointF p) {
            p -= origin;
            return QPointF(match.center.x + c * p.x() - s * p.y(),
                           match.center.y + s * p.x() + c * p.y());
        };
        painter.setPen(QPen(QColor(30, 240, 105), lineWidth * 1.6));
        QPolygonF outline;
        outline << transform({0, 0}) << transform({double(modelSize_.width()), 0})
                << transform({double(modelSize_.width()), double(modelSize_.height())})
                << transform({0, double(modelSize_.height())});
        painter.drawPolygon(outline);
        painter.drawLine(QPointF(match.center.x - 7, match.center.y),
                         QPointF(match.center.x + 7, match.center.y));
        painter.drawLine(QPointF(match.center.x, match.center.y - 7),
                         QPointF(match.center.x, match.center.y + 7));
        painter.setPen(QPen(QColor(255, 110, 70), lineWidth));
        for (const auto& point : modelPoints_)
            painter.drawPoint(transform({point.x, point.y}));
        painter.setPen(QPen(QColor(255, 235, 80), lineWidth));
        painter.drawText(QPointF(match.center.x + 8, match.center.y - 8),
                         QString::number(i + 1));
    }
    painter.restore();
}

void ImageView::mousePressEvent(QMouseEvent* event) {
    if (!selectable_ || image_.isNull() || event->button() != Qt::LeftButton ||
        !imageRect().contains(event->pos())) return;
    dragging_ = true;
    dragStart_ = imagePoint(event->pos());
    selection_ = QRect(dragStart_, dragStart_);
    templatePoints_.clear();
    update();
}

void ImageView::mouseMoveEvent(QMouseEvent* event) {
    if (!dragging_) return;
    selection_ = QRect(dragStart_, imagePoint(event->pos())).normalized();
    update();
}

void ImageView::mouseReleaseEvent(QMouseEvent* event) {
    if (!dragging_ || event->button() != Qt::LeftButton) return;
    dragging_ = false;
    selection_ = QRect(dragStart_, imagePoint(event->pos())).normalized();
    update();
    if (onSelectionChanged_) onSelectionChanged_(selection_);
}
