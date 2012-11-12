
#ifndef PVWIDGETS_PVDATATREEMASKPROXYMODEL_H
#define PVWIDGETS_PVDATATREEMASKPROXYMODEL_H

#include <pvkernel/core/PVDataTreeObject.h>

#include <QAbstractProxyModel>

namespace PVWidgets {

template <class MaskedClass>
class PVDataTreeMaskProxyModel : public QAbstractProxyModel
{
public:
	PVDataTreeMaskProxyModel(QObject *parent = nullptr) :
		QAbstractProxyModel(parent)
	{}

	PVCore::PVDataTreeObjectBase* get_object(QModelIndex const& index) const
	{
		assert(index.isValid());
		return (static_cast<PVCore::PVDataTreeObjectBase*>(index.internalPointer()));
	}

public:
	/* return the count of row used for parent's children
	 */
	virtual int rowCount(const QModelIndex& parent) const
	{
		QAbstractItemModel *src_model = sourceModel();

		QModelIndex src_parent = QModelIndex();

		if(parent.isValid()) {
			src_parent = model_search_internal_pointer(src_model, parent);
		}

		// we have the parent in src_model

		int child_count = src_model->rowCount(src_parent);
		if (child_count == 0) {
			return 0;
		}

		/* We have to check if the children are of class MaskedClass; we test
		 * the first one
		 */
		QModelIndex src_first_child = src_model->index(0, 0, src_parent);
		PVCore::PVDataTreeObjectBase* src_first_child_object = get_object(src_first_child);
		MaskedClass *mc = dynamic_cast<MaskedClass*>(src_first_child_object);

		if (mc == nullptr) {
			// the children are not of class MaskedClass
			return child_count;
		}

		/* the children are of class MaskedClass, we must also count
		 * their children (the grandchildren of parent)
		 */
		int sum = 0;
		for(int i = 0; i < child_count; ++i) {
			QModelIndex child_index = src_model->index(i, src_parent.column(), src_parent);
			sum += src_model->rowCount(child_index);
		}

		return sum;
	}

	/* return the count of column used for parent's children
	 */
	virtual int columnCount(const QModelIndex& index) const
	{
		if (sourceModel() == nullptr) {
			return 0;
		} else {
			return sourceModel()->columnCount(mapToSource(index));
		}
	}

	/* return the parent of index
	 */
	virtual QModelIndex parent(const QModelIndex& index) const
	{
		QAbstractItemModel *src_model = sourceModel();
		if (src_model == nullptr) {
			return QModelIndex();
		}

		if (!index.isValid()) {
			return QModelIndex();
		}

		QModelIndex src_index = model_search_internal_pointer(src_model, index);

		QModelIndex src_parent = src_model->parent(src_index);
		PVCore::PVDataTreeObjectBase* parent_object = get_object(src_parent);
		MaskedClass *mc = dynamic_cast<MaskedClass*>(parent_object);

		if (mc == nullptr) {
			/* the parent is not of class MaskedClass
			 */
			return createIndex(src_parent.row(), src_parent.column(),
			                   src_parent.internalPointer());
		}

		QModelIndex p = src_model->parent(src_parent);
		return createIndex(p.row(), p.column(), p.internalPointer());
	}

	/* return the child of parent indexed by (row, col)
	 */
	virtual QModelIndex index(int row, int col, const QModelIndex& parent) const
	{
		QAbstractItemModel *src_model = sourceModel();

		if (src_model == nullptr) {
			// no source model
			return QModelIndex();
		}

		QModelIndex src_parent = QModelIndex();
		if (parent.isValid()) {
			src_parent = model_search_internal_pointer(src_model, parent);
		}

		int child_count = src_model->rowCount(src_parent);
		if (child_count == 0) {
			// there are no children
			return QModelIndex();
		}

		/* check if the first child has to be masked or not. The first one is tested
		 * instead of the row'th because there is no relation between row and the
		 * src_parent's children count.
		 */
		QModelIndex src_first_child = src_model->index(0, col, src_parent);

		PVCore::PVDataTreeObjectBase* child_object = get_object(src_first_child);
		MaskedClass *mc = dynamic_cast<MaskedClass*>(child_object);

		if (mc == nullptr) {
			// the children do not have to be masked
			QModelIndex src_child = src_model->index(row, col, src_parent);
			return createIndex(row, col, src_child.internalPointer());
		}

		/* the children have to be masked, searching in the grandchildren
		 */
		int rel_row = row;
		for(int i = 0; i < child_count; ++i) {
			QModelIndex child_index =
				src_model->index(i, src_parent.column(), src_parent);
			int count = src_model->rowCount(child_index);

			if (rel_row < count) {
				QModelIndex idx = src_model->index(rel_row, col, child_index);
				return createIndex(row, col, idx.internalPointer());
			}

			rel_row -= count;
		}

		return QModelIndex();
	}

	virtual QModelIndex mapFromSource(QModelIndex const& src_index) const
	{
		if (!src_index.isValid()) {
			return QModelIndex();
		}

		PVCore::PVDataTreeObjectBase* src_object = get_object(src_index);
		MaskedClass *mc = dynamic_cast<MaskedClass*>(src_object);

		if(mc != nullptr) {
			/* the entry is of class MaskedClass; it must be hidden
			 */
			return QModelIndex();
		}

		QModelIndex src_parent = src_index.parent();

		if (!src_parent.isValid()) {
			/* src_parent is the root
			 */
			return createIndex(src_index.row(), src_index.column(), src_index.internalPointer());
		}

		/* at this point, the parent's class must be tested
		 */
		src_object = get_object(src_parent);
		mc = dynamic_cast<MaskedClass*>(src_object);

		if(mc == nullptr) {
			/* the parent is not a MaskedClass
			 */
			return index(src_index.row(), src_index.column(), mapFromSource(src_parent));
		}

		/* the parent is a MaskedClass, src_index must be expressed relatively to its
		 * grand parent
		 */
		QAbstractItemModel *src_model = sourceModel();
		QModelIndex grand_parent = src_parent.parent();
		int row = src_index.row();

		for(int i = 0; i < src_parent.row(); ++i) {
			QModelIndex parent_child_index =
				src_model->index(i, 0, grand_parent);
			row += src_model->rowCount(parent_child_index);
		}

		return index(row, src_index.column(), mapFromSource(grand_parent));
	}

	virtual QModelIndex mapToSource(QModelIndex const& proxy_index) const
	{
		if (!proxy_index.isValid()) {
			return QModelIndex();
		}

		return model_search_internal_pointer(sourceModel(), proxy_index);
	}

	bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole)
	{
		return true;
	}

	QVariant data(const QModelIndex &index, int role) const
	{
		return sourceModel()->data(mapToSource(index), role);
	}

	Qt::ItemFlags flags(const QModelIndex& index) const
	{
		return sourceModel()->flags(mapToSource(index));
	}

private:
	static  QModelIndex model_search_internal_pointer(const QAbstractItemModel *src_model,
	                                                  const QModelIndex &searched,
	                                                  const QModelIndex &parent = QModelIndex())
	{
		if (parent.internalPointer() == searched.internalPointer()) {
			return parent;
		}
		for(int i = 0; i < src_model->rowCount(parent); ++i) {
			QModelIndex child_index = src_model->index(i, 0, parent);
			QModelIndex res = model_search_internal_pointer(src_model, searched, child_index);
			if (res != QModelIndex()) {
				return res;
			}
		}

		return QModelIndex();
	}
};

}

#endif // PVWIDGETS_PVDATATREEMASKPROXYMODEL_H

