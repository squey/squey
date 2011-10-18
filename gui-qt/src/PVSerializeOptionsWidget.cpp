#include "PVSerializeOptionsWidget.h"
#include <QVBoxLayout>

PVInspector::PVSerializeOptionsWidget::PVSerializeOptionsWidget(PVCore::PVSerializeArchiveOptions_p options, QWidget* parent):
	QWidget(parent)
{
	_model = new PVSerializeOptionsModel(options, parent);

	_view = new QTreeView();
	_view->setModel(_model);
	QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);
	sizePolicy.setHorizontalStretch(0);
	sizePolicy.setVerticalStretch(0);
	sizePolicy.setHeightForWidth(_view->sizePolicy().hasHeightForWidth());
	_view->setSizePolicy(sizePolicy);
	_view->setMinimumSize(QSize(0, 250));
	_view->setHeaderHidden(true);

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addWidget(_view);
	setLayout(layout);

	setFocusPolicy(Qt::StrongFocus);
}
