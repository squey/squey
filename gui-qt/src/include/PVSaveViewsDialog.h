#ifndef PVSAVEVIEWSDIALOG_H
#define PVSAVEVIEWSDIALOG_H

#include <picviz/PVView_types.h>

#include <QFileDialog>
#include <QWidget>

namespace PVInspector {

class PVSaveViewsDialog: public QFileDialog
{
public:
	PVSaveViewsDialog(QList<Picviz::PVView_p> const& views, QWidget* parent);

protected:
	QList<Picviz::PVView_p> _views;
};

}

#endif
