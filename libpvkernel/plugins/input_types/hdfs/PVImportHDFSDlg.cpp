#include "PVImportHDFSDlg.h"


PVRush::PVImportHDFSDlg::PVImportHDFSDlg(QStringList const& formats, QWidget* parent):
	QDialog(parent)
{
	setupUi(this);
	_formats->addItems(formats);
}
