//! \file PVAxesCombination.cpp
//! $Id: PVAxesCombination.cpp 3024 2011-06-01 00:29:23Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <assert.h>

#include <pvrush/PVFormat.h>

#include <picviz/PVAxesCombination.h>
#include <picviz/PVAxis.h>
#include <picviz/PVColor.h>

/******************************************************************************
 *
 * Picviz::PVAxesCombination::axis_append
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::axis_append(const PVAxis &axis)
{
	axes_list.push_back(axis);
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::get_axes_count
 *
 *****************************************************************************/
PVCol Picviz::PVAxesCombination::get_axes_count() const
{
	return axes_list.size();
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::get_axes_names_list
 *
 *****************************************************************************/
QStringList Picviz::PVAxesCombination::get_axes_names_list()
{
	int         i;
	QStringList output_list;

	for (i=0; i<axes_list.size(); i++) {
		output_list << axes_list[i].get_name();
	}

	return output_list;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::get_axis
 *
 *****************************************************************************/
const Picviz::PVAxis &Picviz::PVAxesCombination::get_axis(PVCol index) const
{
	assert(!axes_list.empty());

	/* We check that the given axis' index is not out of range */
	if (index >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		return axes_list.last();
	}

	return axes_list[index];
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::get_axis_column_index
 *
 *****************************************************************************/
PVCol Picviz::PVAxesCombination::get_axis_column_index(PVCol index) const
{
	assert(!axes_list.empty());

	/* We check that the given axis' index is not out of range */
	if (index >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		return columns_indexes_list.last();
	}

	return columns_indexes_list[index];
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::get_original_axes_count
 *
 *****************************************************************************/
PVCol Picviz::PVAxesCombination::get_original_axes_count() const
{
	return original_axes_list.size();
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::decrease_axis_column_index
 *
 *****************************************************************************/
bool Picviz::PVAxesCombination::decrease_axis_column_index(PVCol index)
{
	/* We check that the given axis' index is not out of range */
	if (index >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		return true;
	}

	/* If the actual column_index of that axis is not too low, we decrease it */
	if (columns_indexes_list[index] > 0) {
		columns_indexes_list[index] -= 1;
		axes_list[index] = original_axes_list[columns_indexes_list[index]];
	}
	return false;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::increase_axis_column_index
 *
 *****************************************************************************/
bool Picviz::PVAxesCombination::increase_axis_column_index(PVCol index)
{
	if (index >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		return true;
	}

	/* If the actual column_index of that axis is not too high, we increase it */
	if (columns_indexes_list[index] < original_axes_list.size() - 1) {
		columns_indexes_list[index] += 1;
		axes_list[index] = original_axes_list[columns_indexes_list[index]];
	}

	return false;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::move_axis_left_one_position
 *
 *****************************************************************************/
bool Picviz::PVAxesCombination::move_axis_left_one_position(PVCol index)
{
	if (index >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		return true;
	}

	if (index == 0) {
		return false;
	}

	std::swap(axes_list[index], axes_list[index - 1]);
	std::swap(columns_indexes_list[index], columns_indexes_list[index -1]);

	return false;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::move_axis_right_one_position
 *
 *****************************************************************************/
bool Picviz::PVAxesCombination::move_axis_right_one_position(PVCol index)
{
	if (index >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		return true;
	}

	if (index == axes_list.count() - 1) {
		return false;
	}

	std::swap(axes_list[index], axes_list[index + 1]);
	std::swap(columns_indexes_list[index], columns_indexes_list[index + 1]);

	return false;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::move_axis_to_new_position
 *
 *****************************************************************************/
bool Picviz::PVAxesCombination::move_axis_to_new_position(PVCol index_source, PVCol index_dest)
{
	if (index_dest == index_source) {
		return false;
	}
	if (index_source >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index_source, axes_list.size());
		return true;
	}
	if (index_dest >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index_dest, axes_list.size());
		return true;
	}

	if (index_dest > index_source) {
		axes_list.insert(index_dest + 1, axes_list[index_source]);
		columns_indexes_list.insert(index_dest + 1, columns_indexes_list[index_source]);
		axes_list.remove(index_source);
		columns_indexes_list.remove(index_source);
	} else {
		axes_list.insert(index_dest, axes_list[index_source]);
		columns_indexes_list.insert(index_dest, columns_indexes_list[index_source]);
		axes_list.remove(index_source + 1);
		columns_indexes_list.remove(index_source + 1);
	}

	return false;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::remove_axis
 *
 *****************************************************************************/
bool Picviz::PVAxesCombination::remove_axis(PVCol index)
{
	if ( axes_list.size() <= index ) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		return true;
	}

	axes_list.remove(index);
	columns_indexes_list.remove(index);

	return false;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::set_axis_name
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::set_axis_name(PVCol index, const QString &name_)
{
	assert(!axes_list.empty());

	/* We check that the given axis' index is not out of range */
	if (index >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		axes_list.last().set_name(name_);
	}

	axes_list[index].set_name(name_);
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::set_from_format
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::set_from_format(PVRush::PVFormat *format)
{
	float absciss = 0.0f;

	for ( int i = 0; i < format->axes.count(); i++) {
		PVAxis axis;
		PVColor color;
		PVColor titlecolor;

		axis.color.fromQColor(QColor(format->axes[i]["color"]));
		axis.titlecolor.fromQColor(QColor(format->axes[i]["titlecolor"]));
		axis.name = format->axes[i]["name"];
		axis.absciss = absciss;

		abscissae_list.push_back(absciss);
		axes_list.push_back(axis);
		original_axes_list.push_back(axis);
		columns_indexes_list.push_back(i);

		absciss += 1.0f;
	}
}
