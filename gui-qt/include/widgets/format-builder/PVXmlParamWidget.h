/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVXMLPARAMWIDGET_H
#define PVXMLPARAMWIDGET_H
#include <QWidget>
#include <QVBoxLayout>
#include <QVariant>
#include <QLabel>
#include <QModelIndex>
#include <QPushButton>
#include <QDebug>
#include <QPushButton>
#include <QMessageBox>
#include <iostream>
#include <QComboBox>
#include <QDir>
#include <QStringList>
#include <QString>

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamTextEdit.h>
#include <inendi/plugins.h>
#include <PVXmlParamColorDialog.h>
#include <PVXmlParamWidgetBoardAxis.h>
#include <PVXmlParamWidgetBoardFilter.h>
#include <PVXmlParamWidgetBoardSplitterRegEx.h>
//#include <PVXmlDomModel.h>

//#include "include/NodeDom.h"

namespace PVInspector
{
class PVFormatBuilderWidget;

class PVXmlParamWidget : public QWidget
{
	Q_OBJECT

  public:
	enum Type { filterParam, splitterParam, no };

	PVXmlParamWidget(PVFormatBuilderWidget* parent);

	~PVXmlParamWidget() override;

	/**
	 * draw the filter parameter box.
	 * @param nodeFilter
	 */
	void drawForFilter(PVRush::PVXmlTreeNodeDom* nodeFilter);
	/**
	 * clear the parameter box.
	 * @param index
	 * @todo will be delete when all splitter will be plugins
	 */
	void drawForNo(QModelIndex index);
	/**
	 * Draw the regexp parameter box.
	 * @param nodeSplitter
	 * @todo will be delete when all splitter will be plugins
	 */
	void drawForRegEx(PVRush::PVXmlTreeNodeDom* nodeSplitter);
	/**
	 * Draw the regexp parameter box.
	 * @param nodeSplitter
	 */
	void drawForSplitter(PVRush::PVXmlTreeNodeDom* nodeSplitter);
	/**
	 * Draw the converter parameter box.
	 * @param nodeConverter
	 */
	void drawForConverter(PVRush::PVXmlTreeNodeDom* nodeConverter);
	/**
	 * Dras the axis parameter box.
	 * @param nodeOnClick
	 */
	void drawForAxis(PVRush::PVXmlTreeNodeDom* nodeOnClick);

	/**
	 * this is used to update parameter board
	 */
	void addListWidget();
	/**
	 * this is used to update parameter board
	 */
	void removeListWidget();

	/**
	 * get the parameter, form his index
	 * @param i
	 * @return
	 */
	QVariant getParam(int i);

	PVFormatBuilderWidget* parent() { return _parent; }

  private:
	QVBoxLayout* layout;
	Type type;
	QList<QVariant*> listOfParam;
	QList<QWidget*> lesWidgetDuLayout;
	QPushButton* btnApply;
	PVFormatBuilderWidget* _parent;
	QString pluginListURL;
	// bool confirmApply;
	PVRush::PVXmlTreeNodeDom* nodeConfirmApply;
	QModelIndex editingIndex;

  public Q_SLOTS:
	/**
	 * This slot is called when we select a tree item.<br>
	 * He draw the parameter box.
	 * @param index
	 */
	void edit(QModelIndex const& index);

	void slotEmitNeedApply();

	/**
	 * This is onely used for the regex splitter editor, want editing the RegExp
	 * whiche can create or delete some field.
	 */
	void slotForceApply();

	void slotSelectNext();

  Q_SIGNALS:
	void signalNeedApply();
	void signalNeedConfirmApply(QModelIndex&);
	void signalForceApply(QModelIndex&);
	void signalSelectNext();
	void signalQuittingAParamBoard();
};
} // namespace PVInspector

#endif /* PVXMLPARAMWIDGET_H */
