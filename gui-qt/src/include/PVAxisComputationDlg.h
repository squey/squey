#ifndef PVINSPECTOR_PVAXISCOMPUTATION_H
#define PVINSPECTOR_PVAXISCOMPUTATION_H

#include <QDialog>
#include <pvkernel/core/PVArgument.h>

#include <boost/shared_ptr.hpp>

#include "../ui_PVAxisComputationDlg.h"

namespace PVWidgets
{
	class PVArgumentListWidget;
}

namespace Picviz
{
	class PVAxisComputation;
	class PVView;
}

namespace PVInspector {

class PVAxisComputationDlg: public QDialog, Ui::PVAxisComputationDlg
{
	Q_OBJECT
public:
	PVAxisComputationDlg(Picviz::PVView& view, QWidget* parent = NULL);

public:
	boost::shared_ptr<Picviz::PVAxisComputation> get_plugin();

private:
	void init_plugins(QComboBox* cb);
private slots:
	void update_plugin_args();

private:
	boost::shared_ptr<Picviz::PVAxisComputation> _cur_plugin;
	PVCore::PVArgumentList _plugin_args;
};

}

#endif
