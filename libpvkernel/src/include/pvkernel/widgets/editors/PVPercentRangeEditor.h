/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVWIDGETS_PVPERCENTRANGEEDITOR_H
#define PVWIDGETS_PVPERCENTRANGEEDITOR_H

#include <pvkernel/widgets/PVAbstractRangePicker.h>

#include <pvkernel/core/PVPercentRangeType.h>

class QWidget;

namespace PVWidgets
{

/**
 * @class PVPercentRangeEditor
 */
class PVPercentRangeEditor : public PVAbstractRangePicker
{
	Q_OBJECT

	Q_PROPERTY(
	    PVCore::PVPercentRangeType _percent_range_type READ get_values WRITE set_values USER true)

  public:
	/**
	 * Constructor
	 */
	explicit PVPercentRangeEditor(QWidget* parent = nullptr);

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
} // namespace PVWidgets

#endif // PVWIDGETS_PVPERCENTRANGEEDITOR_H
