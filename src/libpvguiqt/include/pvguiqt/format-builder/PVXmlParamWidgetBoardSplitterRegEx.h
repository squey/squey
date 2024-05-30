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

#ifndef PVXMLPARAMWIDGETBOARDSPLITTERREGEX_H
#define PVXMLPARAMWIDGETBOARDSPLITTERREGEX_H
#include <QWidget>
#include <QDir>
#include <QStringList>
#include <QRegExp>
#include <QVBoxLayout>
#include <QLabel>
#include <QVariant>
#include <QDebug>
#include <QGroupBox>
#include <QFrame>
#include <QStyle>
#include <QCheckBox>
#include <QTableWidget>
#include <QFile>
#include <QTableWidgetItem>
#include <QAbstractItemModel>

#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <PVXmlParamWidgetEditorBox.h>
#include <PVXmlParamTextEdit.h>
#include <PVXmlParamComboBox.h>
#include <PVXmlParamColorDialog.h>
namespace App
{

class PVXmlParamWidget;

class PVXmlParamWidgetBoardSplitterRegEx : public QWidget
{
	Q_OBJECT
  public:
	PVXmlParamWidgetBoardSplitterRegEx(PVRush::PVXmlTreeNodeDom* pNode, PVXmlParamWidget* parent);
	~PVXmlParamWidgetBoardSplitterRegEx() override;

	bool confirmAndSave();
	QWidget* getWidgetToFocus();

	void setData(QStringList const& data)
	{
		for (int i = 0; i < data.size(); i++) {
			PVLOG_INFO("(regexp board widget) get data %s\n", qPrintable(data[i]));
		}
		_data = data;

		QString textVal;
		for (int i = 0; i < _data.size(); i++) {
			textVal += _data[i];
			textVal += QChar('\n');
		}

		validWidget->setVal(textVal);
	}

  private:
	void allocBoardFields();
	QVBoxLayout* createTab(const QString& title, QTabWidget* tab);
	void disableConnexion();
	void disAllocBoardFields();
	void draw();
	void initConnexion();
	void initValue();
	void updateHeaderTable();

	// field
	QTabWidget* tabParam;
	PVXmlParamWidgetEditorBox* name;
	PVXmlParamTextEdit* exp;
	PVXmlParamTextEdit* validWidget;
	QCheckBox* checkSaveValidLog;
	QCheckBox* checkUseTable;
	QTableWidget* table;
	QPushButton* openLog;
	QLabel* labelNbr;
	int nbr;
	QPushButton* btnApply;
	bool useTableVerifie();
	PVXmlParamWidget* _parent;

	// editing node
	PVRush::PVXmlTreeNodeDom* node;
	bool flagNeedConfirmAndSave;
	bool flagAskConfirmActivated;
	bool flagSaveRegExpValidator;

  public Q_SLOTS:
	void slotNoteConfirmationNeeded();
	void slotOpenLogValid();
	void slotSaveValidator(bool);
	void slotSetValues();
	void slotSetConfirmedValues();
	void slotShowTable(bool);
	/**
	* Each time we modify the name, we detect if it's a regexp.<br>
	* Then we propose to push it in the regexp field.
	*/
	void slotVerifRegExpInName();
	/**
	* This slot update the table where we can see the selection
	*/
	void slotUpdateTable();
	void regExCount(const QString& reg);
	void exit();

  Q_SIGNALS:
	void signalRefreshView();
	void signalNeedConfirmAndSave();

  protected:
	QStringList _data;
};
} // namespace App
#endif /* PVXMLPARAMWIDGETBOARDSPLITTERREGEX_H */
