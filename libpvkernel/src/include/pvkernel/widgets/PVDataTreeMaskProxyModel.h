
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
	{
		std::cout << "CTOR" << std::endl;
	}

	PVCore::PVDataTreeObjectBase* get_object(QModelIndex const& index) const
	{
		assert(index.isValid());
		return (static_cast<PVCore::PVDataTreeObjectBase*>(index.internalPointer()));
	}

public:
	virtual int rowCount(const QModelIndex& parent) const
	{
		// C'EST FAIT !
		QAbstractItemModel *src_model = sourceModel();

		if(!parent.isValid()) {
			// return the rowCount of the root
			return sourceModel()->rowCount(QModelIndex());
		}

		QModelIndex src_parent = model_search_internal_pointer(parent.internalPointer());
		PVCore::PVDataTreeObjectBase* src_parent_object = get_object(src_parent);
		MaskedClass *mc = dynamic_cast<MaskedClass*>(src_parent_object);

		int child_count = src_model->rowCount(src_parent);

		if (mc == nullptr) {
			return child_count;
		}

		int sum = 0;

		for(int i = 0; i < child_count; ++i) {
			QModelIndex child_index = src_model->index(i, src_parent.column(), src_parent);
			sum += src_model->rowCount(child_index);
		}

		return sum;
	}

	virtual int columnCount(const QModelIndex& index) const
	{
		if (sourceModel() == nullptr) {
			return 0;
		} else {
			return sourceModel()->columnCount(index);
		}
	}

	virtual QModelIndex parent(const QModelIndex& index) const
	{
		std::cout << "::parent" << std::endl;
		if (sourceModel() == nullptr) {
			return QModelIndex();
		}

		QAbstractItemModel *src_model = sourceModel();
		QModelIndex src_index = model_search_internal_pointer(index.internalPointer());

		std::cout << "  src of " << index.row() << " is " << src_index.row() << std::endl;
		QModelIndex src_parent = src_model->parent(src_index);
		PVCore::PVDataTreeObjectBase* parent_object = get_object(src_parent);
		MaskedClass *mc = dynamic_cast<MaskedClass*>(parent_object);

		if (mc == nullptr) {
			return createIndex(src_parent.row(), src_parent.column(),
			                   src_parent.internalPointer());
		}

		QModelIndex p = src_model->parent(src_parent);
		return createIndex(p.row(), p.column(), p.internalPointer());
	}

	virtual QModelIndex index(int row, int col, const QModelIndex& parent) const
	{
		if (sourceModel() == nullptr) {
			return QModelIndex();
		}

		QAbstractItemModel *src_model = sourceModel();
		QModelIndex src_parent = model_search_internal_pointer(parent.internalPointer());

		QModelIndex src_first_child = src_model->index(0, col, src_parent);
		if (!src_first_child.isValid()) {
			return QModelIndex();
		}

		PVCore::PVDataTreeObjectBase* child_object = get_object(src_first_child);
		MaskedClass *mc = dynamic_cast<MaskedClass*>(child_object);

		if (mc == nullptr) {
			return createIndex(row, col, child_object);
		}

		int child_count = src_model->rowCount(src_parent);
		int sum = 0;
		for(int i = 0; i < child_count; ++i) {
			QModelIndex child_index =
				src_model->index(i, src_parent.column(), src_parent);
			sum += src_model->rowCount(child_index);

			if (sum > row) {
				QModelIndex idx = src_model->index(sum-row, col, child_index);
				return createIndex(row, col, idx.internalPointer());
			}
		}

		return QModelIndex();
	}

	virtual QModelIndex mapFromSource(QModelIndex const& src_index) const
	{
		//std::cout << "::mapFromSource for " << proxy_index.row() << std::endl;
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
		std::cout << "::mapToSource for " << proxy_index.row() << std::endl;
		if (!proxy_index.isValid()) {
			std::cout << "  => invalid (because invalid)" << std::endl;
			return QModelIndex();
		}

		PVCore::PVDataTreeObjectBase* proxy_object = get_object(proxy_index);
		MaskedClass *mc = dynamic_cast<MaskedClass*>(proxy_object);

		QModelIndex src_parent = mapToSource(proxy_index.parent());

		if (mc == nullptr) {
			/* the parent is not a MaskedClass
			 */
			std::cout << "  => " << proxy_index.row() << " (not masked)" << std::endl;
			return index(proxy_index.row(), proxy_index.column(),
			             src_parent);
		}

		/* the parent is a MaskedClass, proxy_index must be expressed relatively to its
		 * real parent
		 */
		QAbstractItemModel *src_model = sourceModel();
		int row = proxy_index.row();

		for(int i = 0; i < src_model->rowCount(src_parent); ++i) {
			QModelIndex src_child = src_model->index(i, proxy_index.column(), src_parent);
			int child_nrow = src_model->rowCount(src_child);
			if (row > child_nrow) {
				row -= child_nrow;
				continue;
			}

			std::cout << "  => " << row << " (masked)" << std::endl;
			return index(row, proxy_index.column(), src_child);
		}

		std::cout << "  => invalid (nothing found)" << std::endl;
		return QModelIndex();
	}

private:
	QModelIndex model_search_internal_pointer(const void *ptr,
	                                          const QModelIndex &parent = QModelIndex()) const
	{
		QAbstractItemModel *src_model = sourceModel();
		for(int i = 0; i < src_model->rowCount(parent); ++i) {
			QModelIndex child_index = src_model->index(i, 0, parent);
			if (child_index.internalPointer() == ptr) {
				return child_index;
			}
			model_search_internal_pointer(ptr, child_index);
		}

		return QModelIndex();
	}
};

}

#endif // PVWIDGETS_PVDATATREEMASKPROXYMODEL_H

