#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QImageReader>
#include <QMessageBox>
#include <QTableWidgetItem>

#include <algorithm>
#include <memory>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui_(std::make_unique<Ui::MainWindow>()) {
    ui_->setupUi(this);
    ui_->centralwidget->setSizes({315, 475, 710});
    ui_->resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui_->templateView->setSelectable(true);

    connect(ui_->exampleButton, &QPushButton::clicked, this, [this] { loadRingsExample(); });
    connect(ui_->templateButton, &QPushButton::clicked, this, [this] { browseImage(true); });
    connect(ui_->sceneButton, &QPushButton::clicked, this, [this] { browseImage(false); });
    connect(ui_->maskButton, &QPushButton::clicked, this, [this] { browseMask(); });
    connect(ui_->previousButton, &QPushButton::clicked, this, [this] { stepScene(-1); });
    connect(ui_->nextButton, &QPushButton::clicked, this, [this] { stepScene(1); });
    connect(ui_->createButton, &QPushButton::clicked, this, [this] { buildModel(); });
    connect(ui_->findButton, &QPushButton::clicked, this, [this] { runSearch(); });
    ui_->templateView->setSelectionCallback([this](const QRect& roi) {
        ui_->roiLabel->setText(tr("模板 ROI: x=%1 y=%2 w=%3 h=%4")
                               .arg(roi.x()).arg(roi.y()).arg(roi.width()).arg(roi.height()));
        ui_->templateView->setTemplatePoints({});
    });
    if (QDir("C:/Program Files/MVTec/HALCON-23.05-Progress/examples/images/rings").exists())
        loadRingsExample();
}

MainWindow::~MainWindow() = default;
cv::Mat MainWindow::toGray(const QImage& image) {
    // QImageReader 能读取含中文的路径；克隆像素数据，避免临时 QImage 释放后悬空。
    const QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
    return cv::Mat(gray.height(), gray.width(), CV_8UC1,
                   const_cast<uchar*>(gray.constBits()), gray.bytesPerLine()).clone();
}

bool MainWindow::loadImage(const QString& path, bool isTemplate) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    QImage image = reader.read();
    if (image.isNull()) {
        QMessageBox::warning(this, tr("无法打开图像"), reader.errorString());
        return false;
    }
    if (isTemplate) {
        templateGray_ = toGray(image);
        ui_->templateView->setImage(image);
        ui_->templatePath->setText(path);
        maskGray_.release();
        ui_->maskPath->setText(tr("未使用掩膜"));
        ui_->roiLabel->setText(tr("在模板图中拖动鼠标框选目标"));
    } else {
        sceneGray_ = toGray(image);
        ui_->sceneView->setImage(image);
        ui_->scenePath->setText(path);
        currentScenePath_ = path;
        QDir dir = QFileInfo(path).absoluteDir();
        const QStringList names = dir.entryList({"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"},
                                                QDir::Files, QDir::Name);
        sceneFiles_.clear();
        for (const auto& name : names) sceneFiles_ << dir.absoluteFilePath(name);
        sceneIndex_ = sceneFiles_.indexOf(path);
        ui_->resultsTable->setRowCount(0);
    }
    ui_->statusLabel->setText(tr("已载入 %1 × %2 图像").arg(image.width()).arg(image.height()));
    return true;
}

patmax::Options MainWindow::options() const {
    patmax::Options result;
    result.lowThreshold = ui_->lowSpin->value();
    result.highThreshold = ui_->highSpin->value();
    result.minContrast = ui_->contrastSpin->value();
    result.maxFeatures = ui_->featuresSpin->value();
    result.maxLevels = ui_->levelsSpin->value();
    result.orientationBins = ui_->binsSpin->value();
    result.angleMinDeg = static_cast<float>(ui_->angleMinSpin->value());
    result.angleMaxDeg = static_cast<float>(ui_->angleMaxSpin->value());
    result.angleStepDeg = static_cast<float>(ui_->angleStepSpin->value());
    result.scaleMin = static_cast<float>(ui_->scaleMinSpin->value());
    result.scaleMax = static_cast<float>(ui_->scaleMaxSpin->value());
    result.scaleStep = static_cast<float>(ui_->scaleStepSpin->value());
    result.minScore = static_cast<float>(ui_->minScoreSpin->value());
    result.maxOverlap = static_cast<float>(ui_->maxOverlapSpin->value());
    result.maxMatches = ui_->maxMatchesSpin->value();
    result.polarity = ui_->polarityCombo->currentIndex() == 0 ?
        patmax::Polarity::Signed : patmax::Polarity::IgnoreLocal;
    result.fastRefine = ui_->fastRefineCheck->isChecked();
    return result;
}

bool MainWindow::buildModel() {
    if (templateGray_.empty()) {
        QMessageBox::information(this, tr("缺少模板"), tr("请先打开模板图像。"));
        return false;
    }
    const QRect rect = ui_->templateView->selection();
    const cv::Rect roi(rect.x(), rect.y(), rect.width(), rect.height());
    std::string error;
    if (!matcher_.create(templateGray_, roi, options(), error, maskGray_)) {
        ui_->statusLabel->setText(QString::fromStdString(error));
        QMessageBox::warning(this, tr("建模失败"), QString::fromStdString(error));
        return false;
    }
    // 模型特征是 ROI 内坐标；模板视图需要整张图的像素坐标。
    std::vector<cv::Point2f> points = matcher_.displayPoints();
    for (auto& point : points) point += cv::Point2f(roi.x, roi.y);
    ui_->templateView->setTemplatePoints(points);
    ui_->statusLabel->setText(tr("模型已建立：%1 个边缘特征，%2 层金字塔")
                          .arg(matcher_.featureCount()).arg(matcher_.levelCount()));
    return true;
}

void MainWindow::runSearch() {
    if (sceneGray_.empty()) {
        QMessageBox::information(this, tr("缺少待测图"), tr("请先打开待测图像。"));
        return;
    }
    // 搜索前重新建模，使当前 ROI 和修改后的边缘参数立即生效。
    if (!buildModel()) return;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    ui_->statusLabel->setText(tr("正在搜索…"));
    QApplication::processEvents();
    std::string error;
    patmax::SearchStats stats;
    const auto matches = matcher_.find(sceneGray_, options(), error, &stats);
    QApplication::restoreOverrideCursor();
    if (!error.empty()) {
        ui_->statusLabel->setText(QString::fromStdString(error));
        QMessageBox::warning(this, tr("搜索失败"), QString::fromStdString(error));
        return;
    }
    const cv::Rect roi = matcher_.roi();
    ui_->sceneView->setMatches(matches, QSize(roi.width, roi.height), matcher_.displayPoints());
    ui_->resultsTable->setRowCount(static_cast<int>(matches.size()));
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        const auto& match = matches[i];
        ui_->resultsTable->setItem(i, 0, new QTableWidgetItem(QString::number(i + 1)));
        ui_->resultsTable->setItem(i, 1, new QTableWidgetItem(QString::number(match.center.x, 'f', 1)));
        ui_->resultsTable->setItem(i, 2, new QTableWidgetItem(QString::number(match.center.y, 'f', 1)));
        ui_->resultsTable->setItem(i, 3, new QTableWidgetItem(QString::number(match.angleDeg, 'f', 2)));
        ui_->resultsTable->setItem(i, 4, new QTableWidgetItem(QString::number(match.scale, 'f', 3)));
        ui_->resultsTable->setItem(i, 5, new QTableWidgetItem(QString::number(match.score, 'f', 3)));
    }
    ui_->statusLabel->setText(tr("找到 %1 个；用时 %2 ms（准备 %3、粗搜 %4、细化 %5）\n"
                                 "金字塔 %6 层；候选姿态 %7；候选峰 %8")
        .arg(matches.size()).arg(stats.elapsedMs, 0, 'f', 1)
        .arg(stats.responseMs, 0, 'f', 1).arg(stats.coarseScanMs, 0, 'f', 1)
        .arg(stats.refineMs, 0, 'f', 1).arg(stats.coarseLevel + 1)
        .arg(stats.coarsePoses).arg(stats.coarsePeaks));
}

void MainWindow::browseImage(bool isTemplate) {
    const QString path = QFileDialog::getOpenFileName(this, tr("打开图像"), currentScenePath_,
        tr("图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
    if (!path.isEmpty()) loadImage(path, isTemplate);
}

void MainWindow::browseMask() {
    const QString path = QFileDialog::getOpenFileName(this, tr("打开模板掩膜"), {},
                                                       tr("图像 (*.png *.jpg *.bmp)"));
    if (path.isEmpty()) return;
    QImageReader reader(path);
    const QImage image = reader.read();
    if (image.isNull()) {
        QMessageBox::warning(this, tr("掩膜读取失败"), reader.errorString());
        return;
    }
    maskGray_ = toGray(image);
    ui_->maskPath->setText(path);
}

void MainWindow::loadRingsExample() {
    const QString base = "C:/Program Files/MVTec/HALCON-23.05-Progress/examples/images/rings";
    if (!QDir(base).exists()) {
        QMessageBox::warning(this, tr("示例不存在"), tr("没有找到 HALCON rings 图像目录。"));
        return;
    }
    if (!loadImage(base + "/rings_01.png", true)) return;
    ui_->templateView->setSelection(QRect(330, 110, 210, 210));
    if (loadImage(base + "/rings_06.png", false)) runSearch();
}

void MainWindow::stepScene(int direction) {
    if (sceneFiles_.isEmpty()) return;
    const int next = qBound(0, sceneIndex_ + direction, sceneFiles_.size() - 1);
    if (next != sceneIndex_) loadImage(sceneFiles_[next], false);
}
