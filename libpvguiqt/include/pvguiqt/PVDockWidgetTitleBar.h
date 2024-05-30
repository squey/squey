#ifndef __PVGUIQT_PVDOCKWIDGETTITLEBAR_H__
#define __PVGUIQT_PVDOCKWIDGETTITLEBAR_H__

#include <QWidget>
#include <squey/PVView.h>

class QPushButton;
class QLabel;

namespace PVGuiQt
{

class PVDockWidgetTitleBar : public QWidget
{
    Q_OBJECT;

public:
    PVDockWidgetTitleBar(Squey::PVView* view, QWidget* view_widget, bool has_help_page, QWidget* parent = nullptr);

public:
    void set_help_page_visible(bool visible);
    void set_window_title(const QString& window_title);

protected:
    void mouseMoveEvent(QMouseEvent * event) override;

private:
    void set_button_stylesheet(QPushButton* button, const QString& icon_name);

private Q_SLOTS:
    void set_help_button_stylesheet();
    void set_float_button_stylesheet();
    void set_close_button_stylesheet();

private:
    QLabel* _window_title = nullptr;
    QPushButton* _help_button = nullptr;
    QPushButton* _float_button = nullptr;
    QPushButton* _close_button = nullptr;

};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVDOCKWIDGETTITLEBAR_H__