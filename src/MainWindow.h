#pragma once

#include "ShapeMatcher.h"

#include <QImage>
#include <QMainWindow>
#include <QStringList>
#include <memory>

namespace Ui { class MainWindow; }

class MainWindow : public QMainWindow {
public:
    /// @param parent Qt 父窗口；通常使用默认空指针创建独立主窗口。
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private:
    /// @param image 输入的 Qt 图像。
    /// @return 独立持有像素数据的 8 位单通道 OpenCV 图像。
    static cv::Mat toGray(const QImage& image);
    /// @param path 输入图像路径，支持中文路径。
    /// @param isTemplate 为 true 时载入模板图，否则载入待测图。
    /// @return 读取成功为 true，读取失败时提示错误并返回 false。
    bool loadImage(const QString& path, bool isTemplate);
    /// @brief 按当前 ROI、掩膜及界面参数重新建立模型并显示特征点。
    /// @return 建模成功为 true，失败时提示错误并返回 false。
    bool buildModel();
    /// @brief 搜索当前待测图，更新叠加标记、结果表和耗时信息。
    void runSearch();
    /// @param isTemplate 为 true 时选择模板图，否则选择待测图。
    void browseImage(bool isTemplate);
    /// @brief 选择可选掩膜，供下一次建模使用。
    void browseMask();
    /// @brief 载入 HALCON rings 示例图，设置示例 ROI 并执行搜索。
    void loadRingsExample();
    /// @param direction 相邻图片的移动方向，-1 为上一张，+1 为下一张。
    void stepScene(int direction);
    /// @return 由界面各控件当前值组成的建模与搜索参数。
    patmax::Options options() const;

    std::unique_ptr<Ui::MainWindow> ui_;
    cv::Mat templateGray_;
    cv::Mat sceneGray_;
    cv::Mat maskGray_;
    patmax::ShapeMatcher matcher_;
    QString currentScenePath_;
    QStringList sceneFiles_;
    int sceneIndex_ = -1;
};
