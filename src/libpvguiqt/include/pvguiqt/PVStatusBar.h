
#include <QStatusBar>
#include <QHBoxLayout>
#include <QLabel>

#include <pvbase/general.h>
#include <pvkernel/widgets/PVModdedIcon.h>
#include <pvkernel/widgets/PVMouseButtonsLegend.h>

#ifndef __PVSTATUSBAR_H__
#define __PVSTATUSBAR_H__

namespace PVGuiQt
{

class PVStatusBar : public QStatusBar
{
    Q_OBJECT;

public:
    PVStatusBar(QWidget* parent = nullptr);

public:
    void set_mouse_buttons_legend(const PVWidgets::PVMouseButtonsLegend& legend);
    void clear_mouse_buttons_legend();

    void set_mouse_left_button_legend(QString text);
    void set_mouse_right_button_legend(QString text);
    void set_mouse_scrollwheel_legend(QString text);

private Q_SLOTS:
    void set_gpu_warning_message(PVCore::PVTheme::EColorScheme cs);

private:
    PVWidgets::PVMouseButtonsLegend _default_mouse_buttons_legend;
    QLabel* _mouse_left_label_text = nullptr;
    QLabel* _mouse_right_label_text = nullptr;
    QLabel* _mouse_scrollwheel_label_text = nullptr;
    QLabel* _warning_msg = nullptr;

};

} // namespace PVGuiQt

#endif // __PVSTATUSBAR_H__