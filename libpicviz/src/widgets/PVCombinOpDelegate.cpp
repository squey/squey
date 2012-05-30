
#include <picviz/widgets/PVCombinOpDelegate.h>
#include <pvkernel/core/PVBinaryOperation.h>

#include <QComboBox>

/*****************************************************************************
 * PVWidgets::PVCombinOpDelegate::PVCombinOpDelegate
 *****************************************************************************/

PVWidgets::PVCombinOpDelegate::PVCombinOpDelegate(QObject *parent)
	: QItemDelegate(parent)
{
}

/*****************************************************************************
 * PVWidgets::PVCombinOpDelegate::createEditor
 *****************************************************************************/

QWidget *PVWidgets::PVCombinOpDelegate::createEditor(QWidget *parent,
                                                     const QStyleOptionViewItem &/*option*/,
                                                     const QModelIndex &index) const
{
	if (index.row() == 0) {
		return 0;
	}
	QComboBox *editor = new QComboBox(parent);
	for (int i = (int)PVCore::PVBinaryOperation::FIRST_BINOP; i < (int)PVCore::PVBinaryOperation::LAST_BINOP; ++i) {
		editor->addItem(PVCore::get_binary_operation_name((PVCore::PVBinaryOperation)i),
		                i);
	}
	return editor;
}

/*****************************************************************************
 * PVWidgets::PVCombinOpDelegate::setEditorData
 *****************************************************************************/

void PVWidgets::PVCombinOpDelegate::setEditorData(QWidget *editor,
                                                  const QModelIndex &index) const
{
	int value = index.model()->data(index, Qt::EditRole).toInt();
	QComboBox *combo_box = static_cast<QComboBox*>(editor);
	combo_box->setCurrentIndex(value);
}

/*****************************************************************************
 * PVWidgets::PVCombinOpDelegate::setModelData
 *****************************************************************************/

void PVWidgets::PVCombinOpDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                                 const QModelIndex &index) const
{
	QComboBox *combo_box = static_cast<QComboBox*>(editor);
	int value = combo_box->currentIndex();
	model->setData(index, value, Qt::EditRole);
}

/*****************************************************************************
 * PVWidgets::PVCombinOpDelegate::updateEditorGeometry
 *****************************************************************************/

void PVWidgets::PVCombinOpDelegate::updateEditorGeometry(QWidget *editor,
                                                         const QStyleOptionViewItem &option,
                                                         const QModelIndex &/* index */) const
{
	editor->setGeometry(option.rect);
}
