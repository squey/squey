#include <QString>

#ifndef __PVMOUSEBUTTONSLEGEND_H__
#define __PVMOUSEBUTTONSLEGEND_H__


namespace PVWidgets
{

class PVMouseButtonsLegend
{
public:
    PVMouseButtonsLegend() : PVMouseButtonsLegend("Select", "Context menu", "Scroll"){};
    PVMouseButtonsLegend(QString left, QString right, QString scrollwheel) :
        _left_button(left),
        _right_button(right),
        _scrollwheel(scrollwheel)
    {};

public:
    void set_left_button_legend(QString text) { _left_button = text; }
    void set_right_button_legend(QString text) { _right_button = text; }
    void set_scrollwheel_legend(QString text) { _scrollwheel = text; }

    QString left_button_legend() const { return _left_button; }
    QString right_button_legend() const { return _right_button; }
    QString scrollwheel_button_legend() const { return _scrollwheel; }

private:
    QString _left_button;
    QString _right_button;
    QString _scrollwheel;

};

} // namespace PVWidgets

Q_DECLARE_METATYPE(PVWidgets::PVMouseButtonsLegend);

static bool metatype_mouse_buttons_legend_registered __attribute((unused)) = []() {
	qRegisterMetaType<PVWidgets::PVMouseButtonsLegend>("PVWidgets::PVMouseButtonsLegend");
	return true;
}();

#endif // __PVMOUSEBUTTONSLEGEND_H__