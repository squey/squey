/**
 * \file PVImportSourceToProjectDlg.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVIMPORTSOURCETOPROJECTDLG_H__
#define __PVGUIQT_PVIMPORTSOURCETOPROJECTDLG_H__

#include <QDialog>
#include <QStringList>

namespace PVGuiQt
{

class PVImportSourceToProjectDlg : public QDialog
{
	Q_OBJECT;
public:
	PVImportSourceToProjectDlg(const QStringList & list, int default_index, QWidget* parent = 0);

public:
	int get_project_index() { return _project_index; }

private slots:
	void set_project_index(int index) { _project_index = index; }

private:
	int _project_index = 0;
};

}

#endif /* __PVGUIQT_PVIMPORTSOURCETOPROJECTDLG_H__ */
