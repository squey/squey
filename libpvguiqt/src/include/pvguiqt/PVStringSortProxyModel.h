/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVSTRINGSORTPROXYMODEL_H
#define PVGUIQT_PVSTRINGSORTPROXYMODEL_H

#include <QAbstractProxyModel>
#include <QModelIndex>
#include <QVector>
#include <QTableView>

#include <pvkernel/core/PVSharedBuffer.h>
#include <picviz/PVSortingFunc.h>

#include <boost/date_time/time_duration.hpp>

#include "tbb/task.h"

namespace PVGuiQt {

class PVStringSortProxyModel: public QAbstractProxyModel
{
	Q_OBJECT

public:
	typedef QVector<int> vec_indexes_t;

public:
	// Public interface
	PVStringSortProxyModel(QTableView* view, QObject* parent = nullptr);

	inline vec_indexes_t const& get_proxy_indexes() const { return _vec_sort_m2s; }

	/**
	 * RH: the parameter has no constness because GCC seems to have a
	 * problem with purity and const methods...
	 */
	void set_qt_order_func(const Picviz::PVQtSortingFunc_flesser& lt_t);

	void set_default_qt_order_func();

	// Helper functions for derived classes
protected:
	void invalidate_all();

	void sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes, tbb::task_group_context* ctxt);

	// Function from QAbstractProxyModel to implement
public:
	QModelIndex mapFromSource(QModelIndex const& src_idx) const override;
	QModelIndex mapToSource(QModelIndex const& proxy_idx) const override;
	void setSourceModel(QAbstractItemModel* midel) override;

	// Function from QAbstractItemModel to implement
public:
	/** Sort the model content on the given column
	 * 
	 * @param[in] column : Column to use for sorting
	 * @param[in] order : Sorting order to use
	 *
	 * @note : This function is a Qt required function to make sortFilter work
	 */
	void sort(int column, Qt::SortOrder order) override;
	int rowCount(const QModelIndex& parent) const override;
	int columnCount(const QModelIndex& parent) const override;
	QModelIndex index(int row, int col, const QModelIndex&) const override;
	QModelIndex parent(const QModelIndex& idx) const override;

signals:
	void sort_cancelled_for_column(int column);

private:
	bool reverse_sort_order();
	bool do_sort(int column, Qt::SortOrder order);
	void init_default_sort();
	void reprocess_source();
	void __do_sort(int column, Qt::SortOrder order, tbb::task_group_context* ctxt);
	bool __reverse_sort_order(tbb::task_group_context* ctxt);
	mutable Picviz::PVQtSortingFunc_flesser _qt_lesser_f;

private slots:
	void src_layout_about_changed();
	void src_layout_changed();
	void src_model_about_reset();
	void src_model_reset();

protected:
	/* View and model are fusionned here because Qt require a sort methods with
	 * a specific signature. Because of this constraint, we can't return the
	 * result of the sort function (success or fail) and we have to handle the
	 * view modification in the sort function.
	 */
	QTableView* _view; //!< View representation of the model.

private:
	vec_indexes_t _vec_sort_m2s; //!< map-to-source indexes after sorting
	int _sort_idx; //!< Index of the column used to perform sorting
	Qt::SortOrder _cur_order; //!< Sorting ordering
};

}

#endif
