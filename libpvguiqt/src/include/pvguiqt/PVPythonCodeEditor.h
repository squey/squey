/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef __PVGUIQT_PVPYTHONCODEEDITOR_H__
#define __PVGUIQT_PVPYTHONCODEEDITOR_H__

#include <QTextEdit>

class QMimeData;

namespace PVGuiQt
{

class PVPythonCodeEditor : public QTextEdit
{
public:
    Q_OBJECT

public:
    enum class EThemeType {
        LIGHT,
        DARK
    };

public:
    PVPythonCodeEditor(EThemeType theme_type, QWidget* parent = nullptr);

protected:
    void insertFromMimeData(const QMimeData * source) override;

private:
    static const char* _theme_types_name[];
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVPYTHONCODEEDITOR_H__