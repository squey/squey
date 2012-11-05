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
public:
	PVImportSourceToProjectDlg(const QStringList & list, int default_index, QWidget* parent = 0);
};

}

#endif /* __PVGUIQT_PVIMPORTSOURCETOPROJECTDLG_H__ */
