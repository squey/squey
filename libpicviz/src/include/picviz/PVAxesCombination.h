/**
 * \file PVAxesCombination.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVAXESCOMBINATION_H
#define PICVIZ_PVAXESCOMBINATION_H

#include <QMetaType>
#include <QStringList>
#include <QVector>

#include <functional>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVAxis.h>
#include <picviz/PVAxesCombination_types.h>

namespace Picviz {

/**
* \class PVAxesCombination
*/
class LibPicvizDecl PVAxesCombination {
	friend class PVCore::PVSerializeObject;
public:
	struct axes_comb_id_t
	{
		axes_comb_id_t()
		{
			data.raw = 0;
		}

		axes_comb_id_t(PVCol ai, uint32_t ci)
		{
			data.info.axis = ai;
			data.info.copy_id = ci;
		}

		axes_comb_id_t(const QVariant &v)
		{
			data.raw = v.toULongLong();
		}

		operator QVariant() const
		{
			return QVariant::fromValue((qulonglong)data.raw);
		}

		bool operator ==(const axes_comb_id_t &e) const
		{
			return data.raw == e.data.raw;
		}

		bool operator !=(const axes_comb_id_t &e) const
		{
			return data.raw != e.data.raw;
		}

		PVCol get_axis() const
		{
			return data.info.axis;
		}

		void set_axis(PVCol v)
		{
			data.info.axis = v;
		}

		PVCol get_id() const
		{
			return data.info.copy_id;
		}

		void set_id(uint32_t v)
		{
			data.info.copy_id = v;
		}

		union {
			struct
			{
				uint32_t copy_id;
				PVCol    axis;
			} info;
			uint64_t raw;
		} data;
	} ;

	typedef QVector<PVAxis>          list_axes_t;
	typedef QVector<axes_comb_id_t>  columns_indexes_t;

private:
	QVector<float>    abscissae_list;       //!< Axes positions, such as [0.0, 1.29, 2.5, 4.76]
	list_axes_t       axes_list;            //!< Contains all the used axes
	columns_indexes_t columns_indexes_list; //!< Contains the indices of the axes to place, such as [0,1,3,0]
	list_axes_t       original_axes_list;   //!< All the axes, left as how they were upon loading the format.
	bool              _is_consistent;                  //!< Whether this object is consistent
public:
	PVAxesCombination();

	/**
	* Add an axis to the list of used axes.
	*
	* @todo do not work as it does not update columns_indexes_list !
	* @param axis The axis to add.
	*/
	void axis_append(const PVAxis &axis);

	/**
	* Add an axis to the list of used axes by index in the original list
	*
	* @param axis The axis to add.
	*/
	void axis_append(PVCol org_axis_id);

	/**
	 * Get the object back to its initial state.
	 */
	void clear();

	/**
	* Decrement the column index of an axis.
	*
	* @param index The current index of the axis to change.
	*
	* @return true if an error occured (the index is out of range), false upon success.
	*/
	bool decrease_axis_column_index(PVCol index);

	/**
	* Get the abscissae_list.
	*
	* @return a const pointer to the first absciss.
	*/
	const float *get_abscissae_list() const { return &abscissae_list[0]; }

	/** @name getters
	*  @{
	*/
	/**
	* Get the current number of used axes.
	*
	* @return The current number of used axes.
	*/
	PVCol get_axes_count() const;

	/**
	 * Gets the QStringList of all Axes names according to the current PVAxesCombination
	 *
	 * @return The list of all names of all current axes
	 *
	 */
	QStringList get_axes_names_list() const;
	
	/**
	* Get an axis, from its index in the list of the currently used axes.
	*
	* @param index The index of the axis in the list of the currently used axes.
	*
	* @return The axis we asked for.
	*
	* @note the returned axis can't be modified.
	* @note if the index is out of range, the last axis is returned.
	*/
	const PVAxis& get_axis(PVCol index) const;
	const PVAxis& get_original_axis(PVCol index) const;

	/**
	* Get the current column index of a currently used axes, from its index.
	*
	* @param index The index of the axis in the list of the currently used axes.
	*
	* @return The current column index for this axis.
	*
	* @note initially, the column indices are 0, 1, 2, 3, 4. This order might change if the user
	* moved an axis.
	*
	* @note If the index is out of range, the column index of the last axis is returned.
	*/
	PVCol get_axis_column_index(PVCol index) const;
	// inline const PVCol* get_axis_column_index_buffer() const
	// {
	// 	return &columns_indexes_list.at(0);
	// }

	/**
	* Get the original axis index from its name.
	*
	* @param name The name of the axis
	*
	* @return The original axis index for this axis.
	*
	*/
	PVCol get_original_axis_index_from_name(QString const& name) const;

	/**
	 * Gets the QStringList of all Axes names according to their original combination
	 *
	 * @return The list of all names of all current axes
	 *
	 */
	QStringList get_original_axes_names_list();

	/**
	* Get the current column index of a currently used axes, from its index.
	*
	* @param index The index of the axis in the list of the currently used axes.
	*
	* @return The current column index for this axis.
	*
	* @note This function is the same as #get_axis_column_index but doesn't do any range index checking, and thus is much faster.
	*/
	PVCol get_axis_column_index_fast(PVCol index) const
	{
		return columns_indexes_list[index].get_axis();
	}

	/**
	* Get the current column index of an original axis.
	*
	* @param index The index of the axis in the list of the currently used axes.
	*
	* @return The original column index for this axis.
	*
	* @note This function is the inverse of #get_axis_column_index.
	*/
	PVCol get_combined_axis_column_index(PVCol index) const;

	QList<PVCol> get_combined_axes_columns_indexes(PVCol index) const;

	/**
	* Get the number of original axes.
	*
	* @return The number of original axes.
	*
	* @note This number is constant, and not always the same as the number of currently used axes since
	*       the user might have deleted one or more axes (or, in the future added or duplicated axes).
	*/
	PVCol get_original_axes_count() const;
	/** @} */

	template <class T>
	QList<PVCol> get_original_axes_index_with_tag(T const& tag) const
	{
		QList<PVCol> ret;
		QVector<PVAxis>::const_iterator it;
		PVCol idx = 0;
		for (it = original_axes_list.begin(); it != original_axes_list.end(); it++) {
			if (it->has_tag(tag)) {
				ret.push_back(idx);
			}
			idx++;
		}
		return ret;
	}

	/**
	* Increment the column index of an axis.
	*
	* @param index The current index of the axis to change.
	*
	* @return true if an error occured (the index is out of range), false upon success.
	*
	*/
	bool increase_axis_column_index(PVCol index);

	inline bool is_consistent() const { return _is_consistent; }

	/**
	 * Returns true if the current axes combination is the default one.
	 */
	bool is_default() const;

	bool is_empty() const;

	/**
	* Move one of the used axes to the left.
	*
	* @param index The current index of the axis to move.
	*
	* @return true if an error occured (the index is out of range), false upon success.
	*
	* @note This function does nothing if the axis is already at the leftmost postion.
	*/
	bool move_axis_left_one_position(PVCol index);

	template <class Iterator>
	bool move_axes_left_one_position(Iterator begin, Iterator end);

	/**
	* Move one of the used axes to the right.
	*
	* @param index The current index of the axis to move.
	*
	* @return true if an error occured (the index is out of range), false upon success.
	*
	* @note This function does nothing if the axis is already at the rightmost postion.
	*/
	bool move_axis_right_one_position(PVCol index);

	template <class Iterator>
	bool move_axes_right_one_position(Iterator begin, Iterator end);

	/**
	* Move one of the used axes to a new position.
	*
	* @param index_source The current index of the axis to move.
	* @param index_dest   The index of the axis after the move.
	*
	* @return true if an error occured (the index is out of range), false upon success.
	*
	*/
	bool move_axis_to_new_position(PVCol index_source, PVCol index_dest);

	template <class Iterator>
	bool move_axes_to_new_position(Iterator begin, Iterator end, PVCol index_dest);

	/**
	*
	*/
	bool remove_axis(PVCol index);
	
	template <class L>
	bool remove_axes(L const& list_idx);

	/**
	 * Reset the axis combination to the default one.
	 */
	void reset_to_default();

	/**
	 * Create an axis combination sorted by name
	 * \param[in] order True to sort
	 */
	void sort_by_name(bool order = true);

	/**
	*
	*/
	void set_from_format(PVRush::PVFormat &format);

	/**
	 * Sets the name of the given axis, according to the current positions of axes
	 *
	 * @param index The Index of the targeted PVAXis
	 * @param name_ The new name_
	 *
	 */
	void set_axis_name(PVCol index, const QString &name_);

	/**
	 * @brief Replace original axes.
	 * It will try to keep the existing combination if the axes contained in
	 * `axes' are in `used_axes'.
	 */
	void set_original_axes(PVRush::list_axes_t const& axes);

	/**
	 * @brief Find the first available child_id for an axis
	 */
	uint32_t get_first_free_child_id(PVCol index);

	/**
	 * @brief Get the "id" of the i-th entry
	 */
	inline axes_comb_id_t get_axes_comb_id(PVCol i) const
	{
		assert (i < columns_indexes_list.size());
		return columns_indexes_list[i];
	}

	/**
	 * @brief Get the index of e
	 */
	PVCol get_index_by_id(const axes_comb_id_t &e) const;

	QString to_string() const;

	list_axes_t const& get_original_axes_list() const { return original_axes_list; }
	list_axes_t const& get_axes_list() const { return axes_list; }

	columns_indexes_t const& get_axes_index_list() const { return columns_indexes_list; }
	void set_axes_index_list(columns_indexes_t const& idxes, list_axes_t const& axes)
	{
		assert(idxes.size() == axes.size());
		columns_indexes_list = idxes;
		axes_list = axes;
	}

protected:
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);

	PVSERIALIZEOBJECT_SPLIT
};


template <class Iterator>
bool PVAxesCombination::move_axes_left_one_position(Iterator begin, Iterator end)
{
	std::sort(begin, end);
	bool ret = false;
	Iterator it;
	for (it = begin; it != end; it++) {
		ret |= move_axis_left_one_position(*it);
	}
	return ret;
}

template <class Iterator>
bool PVAxesCombination::move_axes_right_one_position(Iterator begin, Iterator end)
{
	std::sort(begin, end, std::greater<PVCol>());
	bool ret = false;
	Iterator it;
	for (it = begin; it != end; it++) {
		ret |= move_axis_right_one_position(*it);
	}

	return ret;
}

template <class Iterator>
bool PVAxesCombination::move_axes_to_new_position(Iterator begin, Iterator end, PVCol index_dest)
{
	bool ret = false;
	Iterator it;
	for (it = begin; it != end; it++) {
		ret |= move_axis_to_new_position(*it, index_dest);
		index_dest++;
		if (index_dest >= axes_list.size()) {
			index_dest = axes_list.size()-1;
		}
	}

	return ret;
}

template <class L>
bool PVAxesCombination::remove_axes(L const& list_idx)
{
	QVector<PVAxis> tmp_axes;
	columns_indexes_t tmp_col_indexes;
	PVCol new_size = axes_list.size() - list_idx.size();
	tmp_axes.reserve(new_size);
	tmp_col_indexes.reserve(new_size);
	for (PVCol i = 0; i < axes_list.size(); i++) {
		if (std::find(list_idx.begin(), list_idx.end(), i) == list_idx.end()) {
			tmp_axes.push_back(axes_list.at(i));
			tmp_col_indexes.push_back(columns_indexes_list.at(i));
		}
	}
	axes_list = tmp_axes;
	columns_indexes_list = tmp_col_indexes;
	return true;
}

}

Q_DECLARE_METATYPE(Picviz::PVAxesCombination::axes_comb_id_t)

#endif	/* PICVIZ_PVAXESCOMBINATION_H */
