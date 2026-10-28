#include "MainWindow.h"

#include <QApplication>
#include <QTimer>

/**
 * @brief 启动验证界面，并提供可选的自动截图入口。
 * @param argc 命令行参数个数。
 * @param argv 可包含 --ui-smoke 和其后的截图文件路径。
 * @return Qt 事件循环的退出码；截图失败时返回 1。
 */
int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    const QStringList args = app.arguments();
    const int smokeIndex = args.indexOf("--ui-smoke");
    if (smokeIndex >= 0) {
        QTimer::singleShot(350, [&app, &window, args, smokeIndex] {
            bool ok = true;
            if (smokeIndex + 1 < args.size()) ok = window.grab().save(args[smokeIndex + 1]);
            app.exit(ok ? 0 : 1);
        });
    }
    return app.exec();
}
