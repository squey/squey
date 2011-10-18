#include "PVSerializeOptionsWidget.h"
#include <QVBoxLayout>

PVInspector::PVSerializeOptionsWidget::PVSerializeOptionsWidget(PVCore::PVSerializeArchiveOptions_p options, QWidget* parent):
	QWidget(parent)
{
	_model = new PVSerializeOptionsModel(options, parent);

	_view = new QTreeView();
	_view->setModel(_model);

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addWidget(_view);

	setFocusPolicy(Qt::StrongFocus);
}
