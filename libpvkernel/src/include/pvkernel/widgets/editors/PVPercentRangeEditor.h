/**
 * \file PVPercentRangeEditor.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVWIDGETS_PVPERCENTRANGEEDITOR_H
#define PVWIDGETS_PVPERCENTRANGEEDITOR_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVPercentRangeType.h>
#include <pvkernel/widgets/PVAbstractRangePicker.h>

class QWidget;

namespace PVWidgets
{

class PVMainWindow;

/**
 * @class PVPercentRangeEditor
 */
class PVPercentRangeEditor : public PVAbstractRangePicker
{
	Q_OBJECT

	Q_PROPERTY(PVCore::PVPercentRangeType _percent_range_type READ get_values WRITE set_values USER true)

public:
	/**
	 * Constructor
	 */
	PVPercentRangeEditor(QWidget *parent = 0);

public:
	/**
	 * returns the editor's curent value
	 *
	 * @return the percent pair
	 */
	PVCore::PVPercentRangeType get_values() const;

	/**
	 * set the editor's curent value
	 *
	 * @param r the value to use
	 */
	void set_values(const PVCore::PVPercentRangeType& r);
};

}

#endif // PVWIDGETS_PVPERCENTRANGEEDITOR_H
