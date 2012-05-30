#ifndef PICVIZ_PVCOMBINOPDELEGATE_H
#define PICVIZ_PVCOMBINOPDELEGATE_H

#include <QItemDelegate>

namespace PVWidgets {

class PVCombinOpDelegate : public QItemDelegate
{
	Q_OBJECT

public:
	PVCombinOpDelegate(QObject *parent = 0);

	QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
	                      const QModelIndex &index) const;

	void setEditorData(QWidget *editor, const QModelIndex &index) const;
	void setModelData(QWidget *editor, QAbstractItemModel *model,
	                  const QModelIndex &index) const;

	void updateEditorGeometry(QWidget *editor,
	                          const QStyleOptionViewItem &option,
	                          const QModelIndex &index) const;
};

}

#endif // PICVIZ_PVCOMBINOPDELEGATE_H
