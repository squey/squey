/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef __PVGUIQT_PVPYTHONSCRIPTWIDGET_H__
#define __PVGUIQT_PVPYTHONSCRIPTWIDGET_H__

#include <QGroupBox>

class QTextEdit;
class QLineEdit;
class QRadioButton;

namespace PVGuiQt
{

class PVPythonScriptWidget : public QGroupBox
{
    Q_OBJECT
    
public:
    PVPythonScriptWidget(QWidget* parent = nullptr);

public:
    void set_python_script(const QString& python_script, bool is_path, bool disabled);
    QString get_python_script(bool& is_path, bool& disabled) const;

private:
    void notify_python_script_updated();

Q_SIGNALS:
    void python_script_updated(const QString& python_script, bool is_path, bool disabled);

private:
    QRadioButton* _python_script_path_radio;
    QRadioButton* _python_script_content_radio;
    QLineEdit* _exec_python_file_line_edit;
    QTextEdit* _python_script_content_text;
};

} // namespace  PVGuiQt

#endif // __PVGUIQT_PVPYTHONSCRIPTWIDGET_H__