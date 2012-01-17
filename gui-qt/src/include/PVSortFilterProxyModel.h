#ifndef PVINSPECTOR_PVFILTERPROXYMODEL_H
#define PVINSPECTOR_PVFILTERPROXYMODEL_H

#include <QAbstractProxyModel>
#include <QModelIndex>
#include <QVector>

namespace PVInspector {

namespace __impl {
class PVSortProxyAsc;
class PVSortProxyDesc;
}

class PVSortFilterProxyModel: public QAbstractProxyModel
{
	Q_OBJECT

	friend class __impl::PVSortProxyAsc;
	friend class __impl::PVSortProxyDesc;
public:
	typedef QVector<int> vec_indexes_t;

public:
	PVSortFilterProxyModel(QObject* parent = NULL);

	// Public interface
	inline void set_dynamic_sort(bool enable) { _dyn_sort = enable; }
	inline bool dynamic_sort() const { return _dyn_sort; }
	void reset_to_default_ordering();

	// Helper functions for derived classes
protected:
	void invalidate_sort();
	void invalidate_filter();
	void invalidate_all();

	// Function to reimplement
protected:
	/*! \brief Compare function
	 *  This function needs to be reimplemented in order to implement sorting.
	 */
	virtual bool less_than(const QModelIndex &left, const QModelIndex &right) const = 0;

	/*! \brief Filter source indexes.
	 *  This function can be reimplemented to filter a list of source indexes.
	 *  Its default implementation filters out indexes from src_idxes_in according to the return
	 *  value of filter_source_index.
	 */
	virtual void filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out);

	/*! \brief Filter a source index.
	 */
	virtual bool filter_source_index(int /*idx_in*/) { return true; };

	// Function from QAbstractProxyModel to implement
public:
	virtual QModelIndex mapFromSource(QModelIndex const& src_idx) const;
	virtual QModelIndex mapToSource(QModelIndex const& proxy_idx) const;
	virtual void setSourceModel(QAbstractItemModel* midel);

	// Function from QAbstractItemModel to implement
public:
	virtual void sort(int column, Qt::SortOrder order);
	virtual int rowCount(const QModelIndex& parent) const;
	virtual int columnCount(const QModelIndex& parent) const;
	virtual QModelIndex index(int row, int col, const QModelIndex&) const;
	virtual QModelIndex parent(const QModelIndex& idx) const;

private:
	void reverse_sort_order();
	void do_sort(int column, Qt::SortOrder order);
	void do_filter();
	void init_default_sort();
	void reprocess_source();

private slots:
	void src_layout_about_changed();
	void src_layout_changed();
	void src_model_about_reset();
	void src_model_reset();

private:
	vec_indexes_t _vec_sort_m2s; // map-to-source indexes after sorting
	vec_indexes_t _vec_filtered_m2s; // map-to-source indexes after filtering
	int _sort_idx;
	Qt::SortOrder _cur_order;
	bool _dyn_sort;
};

}

#endif
