//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <PVXmlParamWidget.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamComboBox.h>

#include <PVXmlParamTextEdit.h>

/******************************************************************************
 *
 * App::PVXmlParamWidget::PVXmlParamWidget
 *
 *****************************************************************************/
App::PVXmlParamWidget::PVXmlParamWidget(PVFormatBuilderWidget* parent)
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
 * App::PVXmlParamWidget::~PVXmlParamWidget
 *
 *****************************************************************************/
App::PVXmlParamWidget::~PVXmlParamWidget()
{
	layout->deleteLater();
}

/******************************************************************************
 *
 * App::PVXmlParamWidget::drawForNo
 *
 *****************************************************************************/
void App::PVXmlParamWidget::drawForNo(QModelIndex)
{
	PVLOG_DEBUG("App::PVXmlParamWidget::drawForNo\n");
	// confirmApply = false;
	if (type != no) {
		Q_EMIT signalQuittingAParamBoard();
		removeListWidget();
		type = no;
	}
}

/******************************************************************************
 *
 * App::PVXmlParamWidget::drawForAxis
 *
 *****************************************************************************/
void App::PVXmlParamWidget::drawForAxis(PVRush::PVXmlTreeNodeDom* nodeOnClick)
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
 * App::PVXmlParamWidget::drawForFilter
 *
 *****************************************************************************/
void App::PVXmlParamWidget::drawForFilter(PVRush::PVXmlTreeNodeDom* nodeFilter)
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
 * App::PVXmlParamWidget::drawForRegEx
 *
 *****************************************************************************/
void App::PVXmlParamWidget::drawForRegEx(PVRush::PVXmlTreeNodeDom* nodeSplitter)
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
 * App::PVXmlParamWidget::drawForSplitter
 *
 *****************************************************************************/
void App::PVXmlParamWidget::drawForSplitter(PVRush::PVXmlTreeNodeDom* nodeSplitter)
{
	PVLOG_DEBUG("App::PVXmlParamWidget::drawForSplitter\n");
	assert(nodeSplitter);
	assert(nodeSplitter->getSplitterPlugin());
	QWidget* w = nodeSplitter->getSplitterParamWidget();
	if (w != nullptr) {
		PVLOG_DEBUG("App::PVXmlParamWidget->objectName() =%s\n",
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
		PVLOG_DEBUG("App::PVXmlParamWidget: no widget\n");
	}
}

/******************************************************************************
 *
 * App::PVXmlParamWidget::drawForConverter
 *
 *****************************************************************************/
void App::PVXmlParamWidget::drawForConverter(PVRush::PVXmlTreeNodeDom* nodeConverter)
{
	PVLOG_DEBUG("App::PVXmlParamWidget::drawForConverter\n");
	assert(nodeConverter);
	assert(nodeConverter->getConverterPlugin());
	QWidget* w = nodeConverter->getConverterParamWidget();
	if (w != nullptr) {
		PVLOG_DEBUG("App::PVXmlParamWidget->objectName() =%s\n",
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
		PVLOG_DEBUG("App::PVXmlParamWidget: no widget\n");
	}
}

/******************************************************************************
 *
 * App::PVXmlParamWidget::addListWidget
 *
 *****************************************************************************/
void App::PVXmlParamWidget::addListWidget()
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
 * App::PVXmlParamWidget::removeListWidget
 *
 *****************************************************************************/
void App::PVXmlParamWidget::removeListWidget()
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
 * App::PVXmlParamWidget::getParam
 *
 *****************************************************************************/
QVariant App::PVXmlParamWidget::getParam(int i)
{
	int id;
	if (i == 0) {
		id = 1;
	} else if (i == 1) {
		id = 3;
	} else if (i == 2) {
		id = 5;
	} else {
		return {};
	}
	return ((PVXmlParamWidgetEditorBox*)lesWidgetDuLayout.at(id))->val();
}

/******************************** SLOTS ***************************************/

/******************************************************************************
 *
 * App::PVXmlParamWidget::edit
 *
 *****************************************************************************/
void App::PVXmlParamWidget::edit(QModelIndex const& index)
{
	PVLOG_DEBUG("App::PVXmlParamWidget::edit\n");
	drawForNo(index);
	if (index.isValid()) {
		editingIndex = index;
		auto* nodeOnClick = (PVRush::PVXmlTreeNodeDom*)index.internalPointer();

		if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::filter) {
			PVLOG_DEBUG("App::PVXmlParamWidget::edit -> filter\n");
			drawForFilter(nodeOnClick);
		} else if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::RegEx) { //|| (splitter &&
			//(nodeOnClick->attribute("type","") ==
			//"regexp"))
			PVLOG_DEBUG("App::PVXmlParamWidget::edit -> regex\n");
			drawForRegEx(nodeOnClick);
			// confirmApply = false;
		} else if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::axis) {
			PVLOG_DEBUG("App::PVXmlParamWidget::edit -> axis\n");
			drawForAxis(nodeOnClick);
		} else if (nodeOnClick->attribute("type", "") == "url") {
			return;
		} else if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::splitter) {
			PVLOG_DEBUG("App::PVXmlParamWidget::edit -> splitter\n");
			drawForSplitter(nodeOnClick);
		} else if (nodeOnClick->type == PVRush::PVXmlTreeNodeDom::Type::converter) {
			PVLOG_DEBUG("App::PVXmlParamWidget::edit -> converter\n");
			drawForConverter(nodeOnClick);
		}
	}
}

/******************************************************************************
 *
 * App::PVXmlParamWidget::slotForceApply
 *
 *****************************************************************************/
void App::PVXmlParamWidget::slotForceApply()
{
	Q_EMIT signalForceApply(editingIndex);
}

/******************************************************************************
 *
 * App::PVXmlParamWidget::slotEmitNeedApply
 *
 *****************************************************************************/
void App::PVXmlParamWidget::slotEmitNeedApply()
{
	Q_EMIT signalNeedApply();
}

/******************************************************************************
 *
 * App::PVXmlParamWidget::slotSelectNext
 *
 *****************************************************************************/
void App::PVXmlParamWidget::slotSelectNext()
{
	Q_EMIT signalSelectNext();
}
