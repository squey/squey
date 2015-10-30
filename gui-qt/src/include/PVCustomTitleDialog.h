/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCUSTOMTITLEDIALOG_H
#define PVCUSTOMTITLEDIALOG_H

#include <QDialog>

namespace PVInspector {

class PVCustomTitleDialog: public QDialog
{
public:
	PVCustomTitleDialog(QWidget* parent = 0, Qt::WindowFlags f = 0);
protected:
	void paintEvent(QPaintEvent* event);
};

}

#endif
