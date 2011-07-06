//! \file PVArgumentListDelegate.cpp
//! $Id: PVArgumentListDelegate.cpp 3201 2011-06-25 18:23:27Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <PVArgumentEditorCreator.h>
#include <PVArgumentListDelegate.h>

#include <PVAxisIndexEditor.h>
#include <PVAxesIndexEditor.h>
#include <PVRegexpEditor.h>
#include <PVEnumEditor.h>
#include <PVColorGradientDualSliderEditor.h>
#include <PVSpinBoxEditor.h>

PVInspector::PVArgumentListDelegate::PVArgumentListDelegate(Picviz::PVView& view, QTableView* parent) :
	QStyledItemDelegate(parent)
{

	_parent = parent;

	// Create the item factory !
	QItemEditorFactory *factory = new QItemEditorFactory;
	
	// Creator for our arguments
	// TODO: have a list of them somewhere ?
	QItemEditorCreatorBase *pv_axis_index_creator = new PVArgumentEditorCreator<PVAxisIndexEditor>(view);
	QItemEditorCreatorBase *pv_axes_index_creator = new PVArgumentEditorCreator<PVAxesIndexEditor>(view);
	QItemEditorCreatorBase *pv_enum_creator = new PVArgumentEditorCreator<PVEnumEditor>(view);
	QItemEditorCreatorBase *regexp_creator = new PVArgumentEditorCreator<PVRegexpEditor>(view);
	QItemEditorCreatorBase *dualslider_creator = new PVArgumentEditorCreator<PVColorGradientDualSliderEditor>(view);
	QItemEditorCreatorBase *spinbox_creator = new PVArgumentEditorCreator<PVSpinBoxEditor>(view);

	// And register them into the factory
	factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexType>(), pv_axis_index_creator);
	factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxesIndexType>(), pv_axes_index_creator);
	factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVEnumType>(), pv_enum_creator);
	factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVColorGradientDualSliderType>(), dualslider_creator);
	factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVSpinBoxType>(), spinbox_creator);
	factory->registerEditor(QVariant::RegExp, regexp_creator);

	setItemEditorFactory(factory);
}

QWidget* PVInspector::PVArgumentListDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	return QStyledItemDelegate::createEditor(parent, option, index);
}

bool PVInspector::PVArgumentListDelegate::editorEvent(QEvent * event, QAbstractItemModel * model, const QStyleOptionViewItem & option, const QModelIndex & index)
{
	return QStyledItemDelegate::editorEvent(event, model, option, index);
}

void PVInspector::PVArgumentListDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
	QStyledItemDelegate::setEditorData(editor, index);
}

void PVInspector::PVArgumentListDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
	QStyledItemDelegate::setModelData(editor, model, index);
}

void PVInspector::PVArgumentListDelegate::updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem &option, const QModelIndex& /*index*/) const
{
	// qDebug("Editor geometry=%d,%d\n", option.rect.x(), option.rect.y());
	editor->setGeometry(option.rect);
}

QSize PVInspector::PVArgumentListDelegate::sizeHint(const QStyleOptionViewItem & option, const QModelIndex & index) const
{
	if (index.column() == 0) {
		return QStyledItemDelegate::sizeHint(option, index);
	}	

	QWidget* widget = createEditor(NULL, option, index);
	
	setEditorData(widget, index);
	QSize ret = widget->sizeHint();
	widget->deleteLater();
	ret.setWidth(ret.width()*1.2);
	ret.setHeight(ret.height()*1.2);

	// qDebug("sizeHint height: %d; width: %d\n", ret.height(), ret.width());

	return ret;
}

// It seems the QTableView does not care of the delegare sizeHint(), making
// the widget not having a good size. It is a ugly hack but the user get 
// a good feeling :)
QSize PVInspector::PVArgumentListDelegate::getSize() const 
{
	int rows = _parent->model()->rowCount();

	int w = 0;
	int max_w = 0;
	int h;
	QStyleOptionViewItem view_options;

	QModelIndex model_index;
	QWidget *widget;
	QSize ret;

	h = 0;
	for (int r=0; r < rows; r++) {
		w = 0;

		// Discover widget height
		model_index = _parent->model()->index(r, 1);
		widget = createEditor(NULL, view_options, model_index);
		ret = widget->sizeHint();
		h = h + ret.height();

		// Discover widget width
		for (int c=0; c < 2; c++) {
			model_index = _parent->model()->index(r, c);
			widget = createEditor(NULL, view_options, model_index);
			ret = widget->sizeHint();
			w  += ret.width();
		}
		if (w > max_w) {
			max_w = w;
		}
	}

	// 55 is a number that make the widget appear nicely. This is ugly
	// but how can we go around this?
	max_w += 70;

	// For some reason, I cannot get the btn_layout height. So again
	// we uglyly increase the value to make it appear nice.
	h += 150;

	return QSize(max_w, h);
}
// int PVInspector::PVArgumentListDelegate::sizeHintForColumn(int column) const
// {
// 	return 0;
// }

void PVInspector::PVArgumentListDelegate::paint(QPainter* painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
	if (index.column() == 0) {
		return QStyledItemDelegate::paint(painter, option, index);
	}

	_parent->openPersistentEditor(index);

	// painter->save();
	// painter->setRenderHint(QPainter::Antialiasing, true);

	// QWidget *widget = createEditor(NULL, option, index);

	// // widget->setFocus(Qt::OtherFocusReason);

	// widget->resize(option.rect.size());
	// setEditorData(widget, index);
	
	// // _parent->setFocusProxy(widget);
	// widget->setFocusProxy(_parent);
	// widget->setFocusPolicy(Qt::StrongFocus);

	// int x = _parent->pos().x() + option.rect.x();
	// int y = _parent->pos().y() + option.rect.y();

	// widget->render(painter, QPoint(x, y));

	// widget->deleteLater();

	// painter->restore();

	// QWidget* widget = createEditor(NULL, option, index);

	// painter->save();

	// widget->render(painter);

	// // delete the widget
	// widget->deleteLater();

	// painter->restore();


	// // Get the widget from the item factory using createEditor, and disable it
	// // This widget has no parent, as it will only be used for drawing and deleted after

	// QWidget* widget = createEditor(NULL, option, index);
	// // // FIXME: cf. below
	// // QSize s = option.rect.size();
	// // s.setWidth((double)s.width()*0.8);
	// // s.setHeight((double)s.height()*0.8);
	// widget->resize(option.rect.size());
	// setEditorData(widget, index);
	// // // widget->setEnabled(false);

	// // // Draw the widget. Taken from QStyledItemDelegate::paint
	// // //painter->fillRect(option.rect, Qt::black);
	// // //painter->fillRect(QRect(100, 100, 10, 10), Qt::red);
	// // // FIXME: find out why QWidget::render interpret option.rect.topLeft() according to the main window (and not the table view)
	// // // Quick'n'dirty fix: compute the good position

	// // // QRect rect = _parent->frameGeometry();
	// // int x = _parent->pos().x() + option.rect.x() + 1; // Why '+1'? Seems the widget needs some extra padding
	// // int y = _parent->pos().y() + option.rect.y() + 1; // but I don't know a better way right now.

	// // // painter->save();
	// // // painter->end();
	// // widget->render(painter, QPoint(x,y));
	// // connect(widget, SIGNAL(clicked()), this, SLOT(widget_clicked_Slot()));
	// widget->render(painter);

	// // delete the widget
	// widget->deleteLater();
}
