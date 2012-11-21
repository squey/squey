/**
 * \file PVAxesCombination.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <assert.h>

#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVAxesCombination.h>
#include <picviz/PVAxis.h>
#include <pvkernel/core/PVColor.h>

/******************************************************************************
 *
 * Picviz::PVAxesCombination::PVAxesCombination
 *
 *****************************************************************************/

Picviz::PVAxesCombination::PVAxesCombination():
	_is_consistent(true)
{
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::axis_append
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::axis_append(const PVAxis &axis)
{
	original_axes_list.push_back(axis);
	axis_append(original_axes_list.size() - 1);
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::axis_append
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::axis_append(PVCol org_axis_id)
{
	assert(org_axis_id < original_axes_list.size());
	columns_indexes_list.push_back(axes_comb_id_t(org_axis_id,
	                                              get_first_free_child_id(org_axis_id)));
	axes_list.push_back(original_axes_list.at(org_axis_id));
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::clear
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::clear()
{
	axes_list.clear();
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
QStringList Picviz::PVAxesCombination::get_axes_names_list() const
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

const Picviz::PVAxis &Picviz::PVAxesCombination::get_original_axis(PVCol index) const
{
	assert(index < original_axes_list.size());
	return original_axes_list.at(index);
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
	if (index >= columns_indexes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, axes_list.size());
		return columns_indexes_list.last().get_axis();
	}

	return columns_indexes_list[index].get_axis();
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
		return columns_indexes_list.last().get_axis();
	}

	
	for (PVCol i = 0; i < columns_indexes_list.size(); i++) {
		if (columns_indexes_list[i].get_axis() == index) {
			return i;
		}
	}
	// Return the last used axis
	return columns_indexes_list.size() - 1;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::get_combined_axes_columns_indexes
 *
 *****************************************************************************/
QList<PVCol> Picviz::PVAxesCombination::get_combined_axes_columns_indexes(PVCol index) const
{
	assert(!axes_list.empty());
	QList<PVCol> cols_ret;

	/* We check that the given axis' index is not out of range */
	if (index >= original_axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index, original_axes_list.size());
		return cols_ret;
	}

	for (PVCol i = 0; i < columns_indexes_list.size(); i++) {
		if (columns_indexes_list[i].get_axis() == index) {
			cols_ret << i;
		}
	}

	return cols_ret;
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
QStringList Picviz::PVAxesCombination::get_original_axes_names_list() const
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
	assert(index < axes_list.size());

	/* If the actual column_index of that axis is not too low, we decrease it */
	if (columns_indexes_list[index].get_axis() > 0) {
		PVCol axis_index = columns_indexes_list[index].get_axis() - 1;
		columns_indexes_list[index].set_axis(axis_index);
		columns_indexes_list[index].set_id(get_first_free_child_id(axis_index));
		axes_list[index] = original_axes_list[axis_index];
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
	assert(index < axes_list.size());

	/* If the actual column_index of that axis is not too high, we increase it */
	if (columns_indexes_list[index].get_axis() < original_axes_list.size() - 1) {
		PVCol axis_index = columns_indexes_list[index].get_axis() + 1;
		columns_indexes_list[index].set_axis(axis_index);
		columns_indexes_list[index].set_id(get_first_free_child_id(axis_index));
		axes_list[index] = original_axes_list[axis_index];
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
		if (columns_indexes_list[i].get_axis() != i) {
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
		return false;
	}

	if (index == 0) {
		return false;
	}

	std::swap(axes_list[index], axes_list[index - 1]);
	std::swap(columns_indexes_list[index], columns_indexes_list[index -1]);

	return true;
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
		return false;
	}

	if (index == axes_list.count() - 1) {
		return false;
	}

	std::swap(axes_list[index], axes_list[index + 1]);
	std::swap(columns_indexes_list[index], columns_indexes_list[index + 1]);

	return true;
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
		return false;
	}
	if (index_dest >= axes_list.size()) {
		PVLOG_ERROR("%s: Index out of range in %d >= %d\n", __FUNCTION__, index_dest, axes_list.size());
		return false;
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

	return true;
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
		return false;
	}

	axes_list.remove(index);
	columns_indexes_list.remove(index);

	return true;
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::reset_to_default
 *
 *****************************************************************************/
void Picviz::PVAxesCombination::reset_to_default()
{
	_is_consistent = false;
	columns_indexes_list.clear();
	axes_list.clear();

	for (PVCol i = 0; i < original_axes_list.size(); i++) {
		axis_append(i);
	}

	_is_consistent = true;
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

	columns_indexes_t axes_comb;
	if (columns_indexes_list.size() > 0) {
		if (axes_comb.size() == 0) {
			axes_comb = columns_indexes_list;
		}
	}
	else {
		for (PVCol index : format.get_axes_comb()) {
			axes_comb.push_back(axes_comb_id_t(index, get_first_free_child_id(index)));

		}
		if (axes_comb.size() == 0) {
			axes_comb.reserve(axes.size());
			for (PVCol i = 0; i < axes.size(); i++) {
				axes_comb.push_back(axes_comb_id_t(i, 0));
			}
		}
	}

	if (original_axes_list.size() == 0) {
		for (it = axes.begin(); it != axes.end(); it++) {
			PVRush::PVAxisFormat const& axis_format = *it;
			PVAxis axis(axis_format);
			original_axes_list.push_back(axis);
		}

		columns_indexes_list.clear();
		columns_indexes_t::iterator it_comb;
		for (it_comb = axes_comb.begin(); it_comb != axes_comb.end(); it_comb++) {
			PVCol col = it_comb->get_axis();
			if (col < original_axes_list.size()) {
				axis_append(col);
			}
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::set_original_axes
 *
 *****************************************************************************/
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

/******************************************************************************
 *
 * Picviz::PVAxesCombination::get_first_child_id
 *
 *****************************************************************************/
uint32_t Picviz::PVAxesCombination::get_first_free_child_id(PVCol index)
{
	uint32_t id = 0;
	size_t i, size = columns_indexes_list.size();

	while(true) {
		axes_comb_id_t ci(index, id);

		for (i = 0; i < size; ++i) {
			if (columns_indexes_list[i] == ci) {
				break;
			}
		}
		if (i == size) {
			return id;
		}
		++id;
	}
}

/******************************************************************************
 *
 * Picviz::PVAxesCombination::get_index_by_id
 *
 *****************************************************************************/
PVCol Picviz::PVAxesCombination::get_index_by_id(const axes_comb_id_t &e) const
{
	for (int i = 0; i < columns_indexes_list.size(); ++i) {
		if (columns_indexes_list[i] == e) {
			return i;
		}
	}

	return PVCOL_INVALID_VALUE;
}

static bool comp_sort(std::pair<QStringRef, PVCol> const& o1, std::pair<QStringRef, PVCol> const& o2)
{
	return o1.first < o2.first;
}
static bool comp_inv_sort(std::pair<QStringRef, PVCol> const& o1, std::pair<QStringRef, PVCol> const& o2)
{
	return o1.first > o2.first;
}

void Picviz::PVAxesCombination::sort_by_name(bool order)
{
	QStringList names = get_original_axes_names_list();
	std::vector<std::pair<QStringRef, PVCol> > vec_tosort;
	vec_tosort.reserve(names.size());
	for (int i = 0; i < names.size(); i++) {
		vec_tosort.push_back(std::pair<QStringRef, PVCol>(QStringRef(&names.at(i)), i));
	}

	if (order) {
		std::stable_sort(vec_tosort.begin(), vec_tosort.end(), comp_sort);
	}
	else {
		std::stable_sort(vec_tosort.begin(), vec_tosort.end(), comp_inv_sort);
	}

	_is_consistent = false;
	columns_indexes_list.clear();
	axes_list.clear();

	std::vector<std::pair<QStringRef, PVCol> >::const_iterator it;
	for (it = vec_tosort.begin(); it != vec_tosort.end(); it++) {
		axis_append(it->second);
	}

	_is_consistent = true;
}

QString Picviz::PVAxesCombination::to_string() const
{
	if (columns_indexes_list.size() == 0) {
		return QString();
	}

	QString ret;
	for (int i = 0; i < columns_indexes_list.size()-1; i++) {
		ret += QString::number(columns_indexes_list[i].get_axis()) + ",";
	}
	ret += QString::number(columns_indexes_list[columns_indexes_list.size()-1].get_axis());
	PVLOG_DEBUG("(Picviz::PVAxesCombination::to_string) string: %s\n", qPrintable(ret));
	return ret;
}

void Picviz::PVAxesCombination::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	columns_indexes_list.clear();
	so.list_attributes("columns_indexes_list", columns_indexes_list, [=](QVariant const& v) { return axes_comb_id_t::from_qvariant(v); });
	axes_list.clear();
	//so.list("axes_list", axes_list);
	for (axes_comb_id_t id: columns_indexes_list) {
		axes_list.append(original_axes_list.at(id.get_axis()));
	}
}

void Picviz::PVAxesCombination::serialize_write(PVCore::PVSerializeObject& so)
{
	so.list_attributes("columns_indexes_list", columns_indexes_list);
	//so.list("axes_list", axes_list);
}
