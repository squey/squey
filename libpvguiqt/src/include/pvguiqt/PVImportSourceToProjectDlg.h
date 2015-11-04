/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVIMPORTSOURCETOPROJECTDLG_H__
#define __PVGUIQT_PVIMPORTSOURCETOPROJECTDLG_H__

#include <QDialog>
#include <QComboBox>

namespace Picviz {
class PVScene;
class PVRoot;
}

namespace PVGuiQt
{

class PVImportSourceToProjectDlg : public QDialog
{
	Q_OBJECT;
public:
	PVImportSourceToProjectDlg(Picviz::PVRoot const& root, Picviz::PVScene const* sel_scene, QWidget* parent = 0);

public:
	Picviz::PVScene const* get_selected_scene() const;

private:
	QComboBox* _combo_box;
};

}

#endif /* __PVGUIQT_PVIMPORTSOURCETOPROJECTDLG_H__ */
