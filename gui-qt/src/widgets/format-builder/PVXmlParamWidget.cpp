/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <PVXmlParamWidget.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamComboBox.h>

#include <PVXmlParamTextEdit.h>

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::PVXmlParamWidget
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidget::PVXmlParamWidget(PVFormatBuilderWidget* parent)
    : QWidget(), _parent(parent)
{
	layout = new QVBoxLayout();
	setObjectName("PVXmlParamWidget");

	// confirmApply = false;
	// listeDeParam = new QList<QVariant*>();
	removeListWidget();
	type = no;
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::~PVXmlParamWidget
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidget::~PVXmlParamWidget()
{
	layout->deleteLater();
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForNo
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForNo(QModelIndex)
{
	PVLOG_DEBUG("PVInspector::PVXmlParamWidget::drawForNo\n");
	// confirmApply = false;
	if (type != no) {
		Q_EMIT signalQuittingAParamBoard();
		removeListWidget();
		type = no;
	}
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForAxis
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForAxis(PVRush::PVXmlTreeNodeDom* nodeOnClick)
{
	auto axisboard = new PVXmlParamWidgetBoardAxis(nodeOnClick, this);
	lesWidgetDuLayout.push_back(axisboard);
	layout->addWidget(axisboard);
	connect(axisboard, &PVXmlParamWidgetBoardAxis::signalRefreshView, this,
	        &PVXmlParamWidget::slotEmitNeedApply);
	connect(axisboard, &PVXmlParamWidgetBoardAxis::signalSelectNext, this,
	        &PVXmlParamWidget::slotSelectNext);
	type = filterParam;
	// focus on name
	axisboard->getWidgetToFocus()->setFocus();
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForFilter
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForFilter(PVRush::PVXmlTreeNodeDom* nodeFilter)
{

	auto filterboard = new PVXmlParamWidgetBoardFilter(nodeFilter, this);
	lesWidgetDuLayout.push_back(filterboard);
	layout->addWidget(filterboard);
	connect(filterboard, &PVXmlParamWidgetBoardFilter::signalRefreshView, this,
	        &PVXmlParamWidget::slotEmitNeedApply);
	connect(filterboard, &PVXmlParamWidgetBoardFilter::signalEmitNext, this,
	        &PVXmlParamWidget::slotSelectNext);
	type = filterParam;

	// focus name.
	filterboard->getWidgetToFocus()->setFocus();
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForRegEx
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForRegEx(PVRush::PVXmlTreeNodeDom* nodeSplitter)
{
	auto regExpBoard = new PVXmlParamWidgetBoardSplitterRegEx(nodeSplitter, this);
	// AG: yes, that's a saturday morning hack
	regExpBoard->setData(nodeSplitter->getDataForRegexp());
	lesWidgetDuLayout.push_back(regExpBoard);
	layout->addWidget(regExpBoard);
	connect(regExpBoard, &PVXmlParamWidgetBoardSplitterRegEx::signalRefreshView, this,
	        &PVXmlParamWidget::slotEmitNeedApply);
	connect(this, &PVXmlParamWidget::signalQuittingAParamBoard, regExpBoard,
	        &PVXmlParamWidgetBoardSplitterRegEx::exit);

	addListWidget();
	type = filterParam;
	// focus on regexp
	regExpBoard->getWidgetToFocus()->setFocus();
}
/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForSplitter
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForSplitter(PVRush::PVXmlTreeNodeDom* nodeSplitter)
{
	PVLOG_DEBUG("PVInspector::PVXmlParamWidget::drawForSplitter\n");
	assert(nodeSplitter);
	assert(nodeSplitter->getSplitterPlugin());
	QWidget* w = nodeSplitter->getSplitterParamWidget();
	if (w != nullptr) {
		PVLOG_DEBUG("PVInspector::PVXmlParamWidget->objectName() =%s\n",
		            qPrintable(w->objectName()));
		lesWidgetDuLayout.push_back(w);
		layout->addWidget(w);
		addListWidget();
		type = splitterParam;

		connect(nodeSplitter, &PVRush::PVXmlTreeNodeDom::data_changed, this,
		        &PVXmlParamWidget::slotEmitNeedApply);
		// slotEmitNeedApply();
		// focus on regexp
		// w->getWidgetToFocus()->setFocus();
	} else {
		PVLOG_DEBUG("PVInspector::PVXmlParamWidget: no widget\n");
	}
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::drawForConverter
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::drawForConverter(PVRush::PVXmlTreeNodeDom* nodeConverter)
{
	PVLOG_DEBUG("PVInspector::PVXmlParamWidget::drawForConverter\n");
	assert(nodeConverter);
	assert(nodeConverter->getConverterPlugin());
	QWidget* w = nodeConverter->getConverterParamWidget();
	if (w != nullptr) {
		PVLOG_DEBUG("PVInspector::PVXmlParamWidget->objectName() =%s\n",
		            qPrintable(w->objectName()));
		lesWidgetDuLayout.push_back(w);
		layout->addWidget(w);
		addListWidget();
		type = splitterParam;

		connect(nodeConverter, &PVRush::PVXmlTreeNodeDom::data_changed, this,
		        &PVXmlParamWidget::slotEmitNeedApply);
		// slotEmitNeedApply();
		// focus on regexp
		// w->getWidgetToFocus()->setFocus();
	} else {
		PVLOG_DEBUG("PVInspector::PVXmlParamWidget: no widget\n");
	}
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::addListWidget
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::addListWidget()
{
	for (int i = 0; i < lesWidgetDuLayout.count(); i++) {
		if (i == 2) {
			layout->addSpacerItem(
			    new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));
		}
		layout->addWidget(lesWidgetDuLayout.at(i));
	}
	layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::removeListWidget
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::removeListWidget()
{
	/*
	 * Suppression de tous les widgets (textes, btn, editBox...).
	 */
	while (!lesWidgetDuLayout.isEmpty()) {
		QWidget* tmp = lesWidgetDuLayout.front();

		tmp->hide();
		tmp->close();
		layout->removeWidget(tmp);
		lesWidgetDuLayout.removeFirst();
		layout->update();
	}

	/*
	 *  Suppression des items (commes les spacers).
	 */
	for (int i = 0; i < 10; i++) {
		layout->removeItem(layout->itemAt(0));
	}
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::getParam
 *
 *****************************************************************************/
QVariant PVInspector::PVXmlParamWidget::getParam(int i)
{
	int id;
	if (i == 0) {
		id = 1;
	} else if (i == 1) {
		id = 3;
	} else if (i == 2) {
		id = 5;
	} else {
		return QVariant();
	}
	return ((PVXmlParamWidgetEditorBox*)lesWidgetDuLayout.at(id))->val();
}

/******************************** SLOTS ***************************************/

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::edit
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::edit(QModelIndex const& index)
{
	PVLOG_DEBUG("PVInspector::PVXmlParamWidget::edit\n");
	drawForNo(index);
	if (index.isValid()) {
		editingIndex = index;
		PVRush::PVXmlTreeNodeDom* nodeOnClick = (PVRush::PVXmlTreeNodeDom*)index.internalPointer();

		if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::filter) {
			PVLOG_DEBUG("PVInspector::PVXmlParamWidget::edit -> filter\n");
			drawForFilter(nodeOnClick);
		} else if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::RegEx) { //|| (splitter &&
			//(nodeOnClick->attribute("type","") ==
			//"regexp"))
			PVLOG_DEBUG("PVInspector::PVXmlParamWidget::edit -> regex\n");
			drawForRegEx(nodeOnClick);
			// confirmApply = false;
		} else if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::axis) {
			PVLOG_DEBUG("PVInspector::PVXmlParamWidget::edit -> axis\n");
			drawForAxis(nodeOnClick);
		} else if (nodeOnClick->attribute("type", "") == "url") {
			return;
		} else if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::splitter) {
			PVLOG_DEBUG("PVInspector::PVXmlParamWidget::edit -> splitter\n");
			drawForSplitter(nodeOnClick);
		} else if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::converter) {
			PVLOG_DEBUG("PVInspector::PVXmlParamWidget::edit -> converter\n");
			drawForConverter(nodeOnClick);
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::slotForceApply
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::slotForceApply()
{
	Q_EMIT signalForceApply(editingIndex);
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::slotEmitNeedApply
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::slotEmitNeedApply()
{
	Q_EMIT signalNeedApply();
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidget::slotSelectNext
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidget::slotSelectNext()
{
	Q_EMIT signalSelectNext();
}
