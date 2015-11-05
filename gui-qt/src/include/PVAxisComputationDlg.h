/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVINSPECTOR_PVAXISCOMPUTATION_H
#define PVINSPECTOR_PVAXISCOMPUTATION_H

#include <QDialog>
#include <pvkernel/core/PVArgument.h>

#include <memory>

#include <ui_PVAxisComputationDlg.h>

namespace PVWidgets
{
	class PVArgumentListWidget;
}

namespace Inendi
{
	class PVAxisComputation;
	class PVView;
}

namespace PVInspector {

class PVAxisComputationDlg: public QDialog, Ui::PVAxisComputationDlg
{
	Q_OBJECT
public:
	PVAxisComputationDlg(Inendi::PVView& view, QWidget* parent = NULL);

public:
	std::shared_ptr<Inendi::PVAxisComputation> get_plugin();

private:
	void init_plugins(QComboBox* cb);
private slots:
	void update_plugin_args();

private:
	std::shared_ptr<Inendi::PVAxisComputation> _cur_plugin;
	PVCore::PVArgumentList _plugin_args;
};

}

#endif
