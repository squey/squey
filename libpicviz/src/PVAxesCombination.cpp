//! \file PVAxesCombination.cpp
//! $Id: PVAxesCombination.cpp 3024 2011-06-01 00:29:23Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <assert.h>

#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVAxesCombination.h>
#include <picviz/PVAxis.h>
#include <pvkernel/core/PVColor.h>

#define ABSCISSAE_DIFF 1.0f

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
 * Picviz::PVAxesCombination::axis_append
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::axis_append(PVCol org_axis_id)
{
	assert(org_axis_id < original_axes_list.size());
	columns_indexes_list.push_back(org_axis_id);
	axes_list.push_back(original_axes_list.at(org_axis_id));
	float absciss = 0.0f;
	if (abscissae_list.size() > 0) {
		absciss = abscissae_list.back()+ ABSCISSAE_DIFF;
	}
	abscissae_list.push_back(absciss);
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::clear
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::clear()
{
	axes_list.clear();
	abscissae_list.clear();
	columns_indexes_list.clear();
	original_axes_list.clear();
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
 * Picviz::PVAxesCombination::get_combined_axis_column_index
 *
 *****************************************************************************/
PVCol Picviz::PVAxesCombination::get_combined_axis_column_index(PVCol index) const
{
	assert(!axes_list.empty());

	/* We check that the given axis' index is not out of range */
	if (index >= original_axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, original_axes_list.size());
		return columns_indexes_list.last();
	}

	
	for (PVCol i = 0; i < columns_indexes_list.size(); i++) {
		if (columns_indexes_list[i] == index) {
			return i;
		}
	}
	// Return the last used axis
	return columns_indexes_list.size() - 1;
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
 * Picviz::PVAxesCombination::get_original_axes_names_list
 *
 *****************************************************************************/
QStringList Picviz::PVAxesCombination::get_original_axes_names_list()
{
	int         i;
	QStringList output_list;

	for (i=0; i< original_axes_list.size(); i++) {
		output_list << original_axes_list[i].get_name();
	}

	return output_list;
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
 * Picviz::PVAxesCombination::is_default
 *
 *****************************************************************************/
bool Picviz::PVAxesCombination::is_default() const
{
	if (columns_indexes_list.size() != original_axes_list.size()) {
		return false;
	}

	for (PVCol i = 0; i < columns_indexes_list.size(); i++) {
		if (columns_indexes_list[i] != i) {
			return false;
		}
	}

	return true;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::is_empty
 *
 *****************************************************************************/
bool Picviz::PVAxesCombination::is_empty() const
{
	return axes_list.size() == 0;
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
 * Picviz::PVAxesCombination::reset_to_default
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::reset_to_default()
{

	columns_indexes_list.clear();
	axes_list.clear();
	abscissae_list.clear();

	for (PVCol i = 0; i < original_axes_list.size(); i++) {
		axis_append(i);
	}
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
void Picviz::PVAxesCombination::set_from_format(PVRush::PVFormat &format)
{
	PVRush::list_axes_t const& axes = format.get_axes();
	PVRush::list_axes_t::const_iterator it;

	std::vector<PVCol> axes_comb = format.get_axes_comb();
	// Default axes combination
	if (axes_comb.size() == 0) {
		axes_comb.reserve(axes.size());
		for (PVCol i = 0; i < axes.size(); i++) {
			axes_comb.push_back(i);
		}
	}

	clear();
	for (it = axes.begin(); it != axes.end(); it++) {
		PVRush::PVAxisFormat const& axis_format = *it;
		PVAxis axis(axis_format);
		original_axes_list.push_back(axis);
	}

	std::vector<PVCol>::iterator it_comb;
	for (it_comb = axes_comb.begin(); it_comb != axes_comb.end(); it_comb++) {
		axis_append(*it_comb);
	}
}

void Picviz::PVAxesCombination::set_original_axes(PVRush::list_axes_t const& axes)
{
	QVector<PVAxis> old_axes_list = axes_list;
	clear();
	PVRush::list_axes_t::const_iterator it;
	for (it = axes.begin(); it != axes.end(); it++) {
		PVRush::PVAxisFormat const& axis_format = *it;
		PVAxis axis(axis_format);
		original_axes_list.push_back(axis);
	}

	for (int id = 0; id < old_axes_list.size(); id++) {
		PVAxis const& axis = old_axes_list.at(id);
		PVLOG_DEBUG("(Picviz::PVAxesCombination::set_original_axes) axis '%s' has unique id '%d'\n", qPrintable(axis.get_name()), axis.get_unique_id());
		int id_org = original_axes_list.indexOf(axis);
		if (id_org != -1) {
			PVLOG_DEBUG("(Picviz::PVAxesCombination::set_original_axes) id_org: %d, axis '%s', unique id: %d \n", id_org, qPrintable(original_axes_list.at(id_org).get_name()), original_axes_list.at(id_org).get_unique_id());
			// We found it. Set the new id.
			axis_append(id_org);
		}
	}
}

QString Picviz::PVAxesCombination::to_string() const
{
	if (columns_indexes_list.size() == 0) {
		return QString();
	}

	QString ret;
	for (int i = 0; i < columns_indexes_list.size()-1; i++) {
		ret += QString::number(columns_indexes_list[i]) + ",";
	}
	ret += QString::number(columns_indexes_list[columns_indexes_list.size()-1]);
	PVLOG_DEBUG("(Picviz::PVAxesCombination::to_string) string: %s\n", qPrintable(ret));
	return ret;
}
