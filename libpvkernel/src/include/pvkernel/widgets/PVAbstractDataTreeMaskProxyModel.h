
#ifndef PVWIDGETS_PVABSTRACTDATATREEMASKPROXYMODEL_H
#define PVWIDGETS_PVABSTRACTDATATREEMASKPROXYMODEL_H

#include <QAbstractProxyModel>

namespace PVWidgets {

class PVAbstractDataTreeMaskProxyModel : public QAbstractProxyModel
{
	Q_OBJECT

public:
	PVAbstractDataTreeMaskProxyModel(QObject *parent = nullptr) :
		QAbstractProxyModel(parent)
	{}

	virtual void setSourceModel(QAbstractItemModel *src_model)
	{
		beginResetModel();

		QAbstractItemModel *old_src_model = sourceModel();
		if (old_src_model) {
			disconnect(old_src_model,
			           SIGNAL(dataChanged(const QModelIndex &, const QModelIndex &)),
			           this,
			           SLOT(data_changed(const QModelIndex &, const QModelIndex &)));

			disconnect(old_src_model,
			           SIGNAL(modelAboutToBeReset()),
			           this,
			           SLOT(model_about_to_be_reset()));

			QAbstractProxyModel::setSourceModel(nullptr);
		}

		if (src_model) {
			QAbstractProxyModel::setSourceModel(src_model);
			connect(src_model,
			        SIGNAL(dataChanged(const QModelIndex &, const QModelIndex &)),
			        this,
			        SLOT(data_changed(const QModelIndex &, const QModelIndex &)));
			connect(src_model,
			        SIGNAL(modelAboutToBeReset()),
			        this,
			        SLOT(model_about_to_be_reset()));
		}

		endResetModel();
	}

protected slots:
	virtual void data_changed(const QModelIndex &topLeft, const QModelIndex &bottomRight) = 0;
	virtual void model_about_to_be_reset() = 0;

};

}

#endif // PVWIDGETS_PVABSTRACTDATATREEMASKPROXYMODEL_H
