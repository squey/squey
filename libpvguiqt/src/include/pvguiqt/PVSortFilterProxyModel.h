/**
 * \file PVSortFilterProxyModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVFILTERPROXYMODEL_H
#define PVGUIQT_PVFILTERPROXYMODEL_H

#include <QAbstractProxyModel>
#include <QModelIndex>
#include <QVector>
#include <QTableView>

#include <pvkernel/core/PVSharedBuffer.h>

#include <boost/date_time/time_duration.hpp>

#include "tbb/task.h"

namespace PVGuiQt {

namespace __impl {
class PVSortProxyAsc;
class PVSortProxyDesc;
class PVSortProxyComp;
}

class PVSortFilterProxyModel: public QAbstractProxyModel
{
	Q_OBJECT

	friend class __impl::PVSortProxyAsc;
	friend class __impl::PVSortProxyDesc;
	friend class __impl::PVSortProxyComp;
public:
	typedef PVCore::PVSharedBuffer<int> vec_indexes_t;

public:
	PVSortFilterProxyModel(QTableView* view, QObject* parent = NULL);

	// Public interface
	inline void set_dynamic_sort(bool enable) { _dyn_sort = enable; }
	inline bool dynamic_sort() const { return _dyn_sort; }
	void reset_to_default_ordering();
	void reset_to_default_ordering_or_reverse();
	inline vec_indexes_t const& get_proxy_indexes() const { return _vec_filtered_m2s; }

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

	/*! \biref Equal compare function
	 *  This function needs to be reimplemented in order to implement sorting.
	 *  It returns true iif left == right.
	 */
	virtual bool is_equal(const QModelIndex &left, const QModelIndex &right) const = 0;

	/*! \brief Global sorting function
	 *  Global sorting function to implement in order to sot directly the array of indexes.
	 *  Default implementation uses the values returned by lees_than.
	 */
	virtual void sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes, tbb::task_group_context* ctxt = NULL);

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

signals:
	void sort_cancelled_for_column(int column);

private:
	void reverse_sort_order();
	void do_sort(int column, Qt::SortOrder order);
	void do_filter();
	void init_default_sort();
	void reprocess_source();
	void __do_sort(int column, Qt::SortOrder order, tbb::task_group_context* ctxt = NULL);
	bool __reverse_sort_order();

private slots:
	void src_layout_about_changed();
	void src_layout_changed();
	void src_model_about_reset();
	void src_model_reset();

protected:
	QTableView* _view;

private:
	vec_indexes_t _vec_sort_m2s; // map-to-source indexes after sorting
	vec_indexes_t _vec_filtered_m2s; // map-to-source indexes after filtering
	int _sort_idx;
	Qt::SortOrder _cur_order;
	bool _dyn_sort;
	double _sort_time;
};

}

#endif
