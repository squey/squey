/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVAXESCOMBINATION_H
#define INENDI_PVAXESCOMBINATION_H

#include <QMetaType>
#include <QStringList>
#include <QVector>

#include <functional>

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVColumnIndexes.h>
#include <pvkernel/rush/PVFormat.h>

#include <inendi/PVAxis.h>

namespace Inendi
{

///**
//* \class PVAxesCombination
//*/
// class PVAxesCombination
//{
//	friend class PVCore::PVSerializeObject;
//
//  public:
//	struct axes_comb_id_t {
//		axes_comb_id_t() { data.raw = 0; }
//
//		axes_comb_id_t(PVCol ai, uint32_t ci)
//		{
//			data.info.axis = ai;
//			data.info.copy_id = ci;
//		}
//
//		axes_comb_id_t(const QVariant& v) { data.raw = v.toULongLong(); }
//
//		operator QVariant() const { return QVariant::fromValue((qulonglong)data.raw); }
//
//		bool operator==(const axes_comb_id_t& e) const { return data.raw == e.data.raw; }
//
//		bool operator<(const axes_comb_id_t& e) const { return data.raw < e.data.raw; }
//
//		bool operator!=(const axes_comb_id_t& e) const { return data.raw != e.data.raw; }
//
//		PVCol get_axis() const { return data.info.axis; }
//
//		void set_axis(PVCol v) { data.info.axis = v; }
//
//		PVCol get_id() const { return data.info.copy_id; }
//
//		void set_id(uint32_t v) { data.info.copy_id = v; }
//
//		static inline axes_comb_id_t from_qvariant(QVariant const& v)
//		{
//			axes_comb_id_t ret;
//			ret.data.raw = (uint64_t)v.toULongLong(nullptr);
//			return ret;
//		}
//
//		union {
//			struct {
//				PVCol axis;
//				uint32_t copy_id;
//			} info;
//			uint64_t raw;
//		} data;
//	};
//
//	typedef QVector<PVAxis> list_axes_t;
//	typedef QVector<axes_comb_id_t> columns_indexes_t;
//
//  private:
//	list_axes_t axes_list; //!< Contains all the used axes
//	columns_indexes_t
//	    columns_indexes_list; //!< Contains the indices of the axes to place, such as [0,1,3,0]
//	list_axes_t
//	    original_axes_list; //!< All the axes, left as how they were upon loading the format.
//  public:
//	PVAxesCombination(PVRush::PVFormat const&);
//	PVAxesCombination(){};
//
//	/**
//	* Add an axis to the list of used axes.
//	*
//	* @todo do not work as it does not update columns_indexes_list !
//	* @param axis The axis to add.
//	*/
//	void axis_append(const PVAxis& axis);
//
//	/**
//	* Add an axis to the list of used axes by index in the original list
//	*
//	* @param axis The axis to add.
//	*/
//	void axis_append(PVCol org_axis_id);
//
//	/**
//	 * Get the object back to its initial state.
//	 */
//	void clear();
//
//	/**
//	* Get the current number of used axes.
//	*
//	* @return The current number of used axes.
//	*/
//	PVCol get_axes_count() const;
//
//	/**
//	 * Gets the QStringList of all Axes names according to the current PVAxesCombination
//	 *
//	 * @return The list of all names of all current axes
//	 *
//	 */
//	QStringList get_axes_names_list() const;
//
//	/**
//	* Get an axis, from its index in the list of the currently used axes.
//	*
//	* @param index The index of the axis in the list of the currently used axes.
//	*
//	* @return The axis we asked for.
//	*
//	* @note the returned axis can't be modified.
//	* @note if the index is out of range, the last axis is returned.
//	*/
//	const PVAxis& get_axis(PVCol index) const;
//	const PVAxis& get_original_axis(PVCol index) const;
//
//	/**
//	* Get the current column index of a currently used axes, from its index.
//	*
//	* @param index The index of the axis in the list of the currently used axes.
//	*
//	* @return The current column index for this axis.
//	*
//	* @note initially, the column indices are 0, 1, 2, 3, 4. This order might change if the user
//	* moved an axis.
//	*
//	* @note If the index is out of range, the column index of the last axis is returned.
//	*/
//	PVCol get_axis_column_index(PVCol index) const;
//
//	/**
//	 * Gets the QStringList of all Axes names according to their original combination
//	 *
//	 * @return The list of all names of all current axes
//	 *
//	 */
//	QStringList get_original_axes_names_list() const;
//
//	/**
//	* Get the current column index of a currently used axes, from its index.
//	*
//	* @param index The index of the axis in the list of the currently used axes.
//	*
//	* @return The current column index for this axis.
//	*
//	* @note This function is the same as #get_axis_column_index but doesn't do any range index
//	*checking, and thus is much faster.
//	*/
//	PVCol get_axis_column_index_fast(PVCol index) const
//	{
//		return columns_indexes_list[index].get_axis();
//	}
//
//	QList<PVCol> get_combined_axes_columns_indexes(PVCol index) const;
//
//	/**
//	* Get the number of original axes.
//	*
//	* @return The number of original axes.
//	*
//	* @note This number is constant, and not always the same as the number of currently used axes
//	*since
//	*       the user might have deleted one or more axes (or, in the future added or duplicated
//	*axes).
//	*/
//	PVCol get_original_axes_count() const;
//
//	/**
//	 * Returns true if the current axes combination is the default one.
//	 */
//	bool is_default() const;
//
//	bool is_empty() const;
//
//	/**
//	* Move one of the used axes to the left.
//	*
//	* @param index The current index of the axis to move.
//	*
//	* @return true if an error occured (the index is out of range), false upon success.
//	*
//	* @note This function does nothing if the axis is already at the leftmost postion.
//	*/
//	bool move_axis_left_one_position(PVCol index);
//
//	template <class Iterator>
//	bool move_axes_left_one_position(Iterator begin, Iterator end);
//
//	/**
//	* Move one of the used axes to the right.
//	*
//	* @param index The current index of the axis to move.
//	*
//	* @return true if an error occured (the index is out of range), false upon success.
//	*
//	* @note This function does nothing if the axis is already at the rightmost postion.
//	*/
//	bool move_axis_right_one_position(PVCol index);
//
//	template <class Iterator>
//	bool move_axes_right_one_position(Iterator begin, Iterator end);
//
//	template <class Iterator>
//	bool move_axes_to_new_position(Iterator begin, Iterator end, PVCol index_dest);
//
//	/**
//	*
//	*/
//	bool remove_axis(PVCol index);
//
//	template <class L>
//	bool remove_axes(L const& list_idx);
//
//	/**
//	 * Reset the axis combination to the default one.
//	 */
//	void reset_to_default();
//
//	/**
//	 * Create an axis combination sorted by name
//	 * \param[in] order True to sort
//	 */
//	void sort_by_name(bool order = true);
//
//	/**
//	*
//	*/
//	void set_from_format(PVRush::PVFormat const& format);
//
//	/**
//	 * @brief Replace original axes.
//	 * It will try to keep the existing combination if the axes contained in
//	 * `axes' are in `used_axes'.
//	 */
//	void set_original_axes(QList<PVRush::PVAxisFormat> const& axes);
//
//	/**
//	 * @brief Find the first available child_id for an axis
//	 */
//	uint32_t get_first_free_child_id(PVCol index);
//
//	/**
//	 * @brief Get the "id" of the i-th entry
//	 */
//	inline axes_comb_id_t get_axes_comb_id(PVCol i) const
//	{
//		assert(i < columns_indexes_list.size());
//		return columns_indexes_list[i];
//	}
//
//	/**
//	 * @brief Get the index of e
//	 */
//	PVCol get_index_by_id(const axes_comb_id_t& e) const;
//	inline bool is_last_axis(const axes_comb_id_t& e) const
//	{
//		return get_index_by_id(e) == get_axes_count() - 1;
//	}
//
//	QString to_string() const;
//
//	list_axes_t const& get_original_axes_list() const { return original_axes_list; }
//	list_axes_t const& get_axes_list() const { return axes_list; }
//
//	columns_indexes_t const& get_axes_index_list() const { return columns_indexes_list; }
//	void set_axes_index_list(columns_indexes_t const& idxes, list_axes_t const& axes)
//	{
//		assert(idxes.size() == axes.size());
//		columns_indexes_list = idxes;
//		axes_list = axes;
//	}
//
//  protected:
//	void serialize_read(PVCore::PVSerializeObject& so);
//	void serialize_write(PVCore::PVSerializeObject& so);
//
//	PVSERIALIZEOBJECT_SPLIT
//};
//
// template <class Iterator>
// bool PVAxesCombination::move_axes_left_one_position(Iterator begin, Iterator end)
//{
//	std::sort(begin, end);
//	bool ret = false;
//	Iterator it;
//	for (it = begin; it != end; it++) {
//		ret |= move_axis_left_one_position(*it);
//	}
//	return ret;
//}
//
// template <class Iterator>
// bool PVAxesCombination::move_axes_right_one_position(Iterator begin, Iterator end)
//{
//	std::sort(begin, end, std::greater<PVCol>());
//	bool ret = false;
//	Iterator it;
//	for (it = begin; it != end; it++) {
//		ret |= move_axis_right_one_position(*it);
//	}
//
//	return ret;
//}
//
// template <class L>
// bool PVAxesCombination::remove_axes(L const& list_idx)
//{
//	QVector<PVAxis> tmp_axes;
//	columns_indexes_t tmp_col_indexes;
//	PVCol new_size = axes_list.size() - list_idx.size();
//	tmp_axes.reserve(new_size);
//	tmp_col_indexes.reserve(new_size);
//	for (PVCol i = 0; i < axes_list.size(); i++) {
//		if (std::find(list_idx.begin(), list_idx.end(), i) == list_idx.end()) {
//			tmp_axes.push_back(axes_list.at(i));
//			tmp_col_indexes.push_back(columns_indexes_list.at(i));
//		}
//	}
//	axes_list = tmp_axes;
//	columns_indexes_list = tmp_col_indexes;
//	return true;
//}

class PVAxesCombination
{
  public:
	PVAxesCombination(PVRush::PVFormat const& format)
	    : _axes(format.get_axes()), _axes_comb(format.get_axes_comb())
	{
	}
	PVAxesCombination(QList<PVRush::PVAxisFormat> const& axes)
	    : _axes(axes), _axes_comb(axes.size())
	{
		std::iota(_axes_comb.begin(), _axes_comb.end(), 0);
	}

  public:
	PVRush::PVAxisFormat const& get_axis(size_t col) const { return _axes[_axes_comb[col]]; }
	PVCol get_nraw_axis(size_t col) const { return _axes_comb[col]; }

	std::vector<PVCol> const& get_combination() const { return _axes_comb; }

	QStringList get_nraw_names() const
	{
		QStringList l;
		for (PVRush::PVAxisFormat const& fmt : _axes) {
			l << fmt.get_name();
		}
		return l;
	}

	QStringList get_combined_names() const
	{
		QStringList l;
		for (PVCol c : _axes_comb) {
			l << get_axis(c).get_name();
		}
		return l;
	}

	size_t get_axes_count() const { return _axes_comb.size(); }

	PVCol get_first_comb_col(PVCol nraw_col) const
	{
		auto it = std::find(_axes_comb.begin(), _axes_comb.end(), nraw_col);
		if (it == _axes_comb.end()) {
			return PVCOL_INVALID_VALUE;
		}

		return std::distance(_axes_comb.begin(), it);
	}

  public:
	void set_combination(std::vector<PVCol> const& comb) { _axes_comb = comb; }

	void axis_append(PVCol comb_col) { _axes_comb.push_back(comb_col); }

	template <class It>
	void move_axes_left_one_position(It&& begin, It const& end)
	{
		for (auto it = begin; it != end; ++it) {
			std::swap(_axes_comb[*it], _axes_comb[*(it - 1)]);
		}
	}

	template <class It>
	void move_axes_right_one_position(It const& begin, It&& end)
	{
		for (auto it = end; it != begin; --it) {
			std::swap(_axes_comb[*it], _axes_comb[*(it - 1)]);
		}
	}

	template <class It>
	void remove_axes(It const& begin, It&& end)
	{
		for (auto it = end - 1; it != begin - 1; --it) {
			_axes_comb.erase(_axes_comb.begin() + *it);
		}
	}

	void reset_to_default()
	{
		_axes_comb.resize(get_axes_count());
		std::iota(_axes_comb.begin(), _axes_comb.end(), 0);
	}

	bool is_default()
	{
		std::vector<PVCol> to_cmp(_axes.size());
		std::iota(to_cmp.begin(), to_cmp.end(), 0);
		return to_cmp == _axes_comb;
	}

	void sort_by_name()
	{
		std::sort(_axes_comb.begin(), _axes_comb.end(), [this](PVCol c1, PVCol c2) {
			return _axes[c1].get_name() < _axes[c2].get_name();
		});
	}

	QString to_string() const
	{
		QStringList res;
		for (PVCol c : _axes_comb) {
			res << QString::number(c);
		}
		return res.join(",");
	}

  private:
	QList<PVRush::PVAxisFormat> const& _axes; //!< View from the PVFormat
	std::vector<PVCol> _axes_comb;

  protected:
	// TODO : Implement this
	friend class PVCore::PVSerializeObject;
	void serialize_read(PVCore::PVSerializeObject&){};
	void serialize_write(PVCore::PVSerializeObject&){};

	PVSERIALIZEOBJECT_SPLIT
};
}

#endif /* INENDI_PVAXESCOMBINATION_H */
