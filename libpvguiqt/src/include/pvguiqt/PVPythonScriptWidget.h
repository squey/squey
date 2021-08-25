/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
