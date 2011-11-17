#ifndef WIDGETS_PVMAPPINGMODEWIDGET_H
#define WIDGETS_PVMAPPINGMODEWIDGET_H

#include <pvkernel/core/general.h>
#include <picviz/PVMapping.h>
#include <picviz/PVView_types.h>
#include <PVComboBox.h>

namespace PVInspector {

namespace PVWidgetsHelpers {

class PVMappingModeWidget: public PVComboBox
{
public:
	PVMappingModeWidget(QWidget* parent = NULL):
		PVComboBox(parent)
	{ }
	PVMappingModeWidget(QString const& type, QWidget* parent = NULL);
	PVMappingModeWidget(PVCol axis_id, Picviz::PVMapping const& mapping, QWidget* parent = NULL);
	PVMappingModeWidget(PVCol axis_id, Picviz::PVView const& view, QWidget* parent = NULL);

public:
	void populate_from_type(QString const& type);
	void populate_from_mapping(PVCol axis_id, Picviz::PVMapping const& mapping);
	inline void select_default() { select("default"); }

public:
	bool set_mode(QString const& mode) { return select_userdata(mode); }
	inline QString get_mode() const { return get_sel_userdata().toString(); }
};

}

}

#endif
