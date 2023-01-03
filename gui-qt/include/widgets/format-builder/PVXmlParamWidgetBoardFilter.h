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

#ifndef PVXMLPARAMWIDGETBOARDFILTER_H
#define PVXMLPARAMWIDGETBOARDFILTER_H
#include <QWidget>
#include <QDir>
#include <QStringList>
#include <QRegExp>
#include <QVBoxLayout>
#include <QLabel>
#include <QVariant>
#include <QDebug>

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamTextEdit.h>
#include <PVXmlParamComboBox.h>
#include <PVXmlParamColorDialog.h>

namespace PVInspector
{

class PVXmlParamWidget;

class PVXmlParamWidgetBoardFilter : public QWidget
{
	Q_OBJECT
  public:
	PVXmlParamWidgetBoardFilter(PVRush::PVXmlTreeNodeDom* node, PVXmlParamWidget* parent);
	~PVXmlParamWidgetBoardFilter() override;
	QWidget* getWidgetToFocus();

  private:
	void allocBoardFields();
	void disableConnexion();
	void disAllocBoardFields();
	void draw();
	void initConnexion();
	void initValue();

	// field
	PVXmlParamWidgetEditorBox* name;
	PVXmlParamTextEdit* exp;
	PVXmlParamTextEdit* validWidget;
	PVXmlParamComboBox* typeOfFilter;
	QPushButton* buttonNext;

	// editing node
	PVRush::PVXmlTreeNodeDom* node;

	PVXmlParamWidget* _parent;

  public Q_SLOTS:
	void slotSetValues();
	void slotVerifRegExpInName();
	void slotEmitNext();

  Q_SIGNALS:
	void signalRefreshView();
	void signalEmitNext();
};
} // namespace PVInspector
#endif /* PVXMLPARAMWIDGETBOARDFILTER_H */
