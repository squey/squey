#ifndef WIDGETS_PVMAPPINGMODEWIDGET_H
#define WIDGETS_PVMAPPINGMODEWIDGET_H

#include <pvkernel/core/general.h>
#include <picviz/PVMapping.h>
#include <picviz/PVView_types.h>
#include <PVComboBox.h>

#include <QPushButton>
#include <QWidget>

namespace PVInspector {

namespace PVWidgetsHelpers {

class PVMappingModeWidget: public QWidget
{
public:
	PVMappingModeWidget(QWidget* parent = NULL):
		QWidget(parent)
	{ init(); }
	PVMappingModeWidget(QString const& type, QWidget* parent = NULL);
	PVMappingModeWidget(PVCol axis_id, Picviz::PVMapping const& mapping, QWidget* parent = NULL);
	PVMappingModeWidget(PVCol axis_id, Picviz::PVView const& view, QWidget* parent = NULL);

public:
	void populate_from_type(QString const& type);
	void populate_from_mapping(PVCol axis_id, Picviz::PVMapping const& mapping);
	inline void select_default() { set_mode("default"); }
	inline void clear() { _combo->clear(); }

	virtual QSize sizeHint() const;

public:
	bool set_mode(QString const& mode) { return _combo->select_userdata(mode); }
	inline QString get_mode() const { return _combo->get_sel_userdata().toString(); }

private:
	void init();

private:
	PVComboBox* _combo;
	QPushButton* _params_btn;
};

}

}

#endif
