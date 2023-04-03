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

#include <PVXmlParamWidgetBoardSplitterRegEx.h>
#include <pvkernel/widgets/PVFileDialog.h>

#define dbg                                                                                        \
	{                                                                                              \
		qDebug() << __FILE__ << ":" << __LINE__;                                                   \
	}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardSplitterRegEx::PVXmlParamWidgetBoardSplitterRegEx
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardSplitterRegEx::PVXmlParamWidgetBoardSplitterRegEx(
    PVRush::PVXmlTreeNodeDom* pNode, PVXmlParamWidget* parent)
    : QWidget(), _parent(parent)
{
	node = pNode;
	allocBoardFields();
	draw();
	flagSaveRegExpValidator = false;
	initValue();
	initConnexion();
	validWidget->setRegEx(exp->toPlainText());
	flagNeedConfirmAndSave = false;
	flagAskConfirmActivated = true;
	setObjectName("PVXmlParamWidgetBoardSplitterRegEx");
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardSplitterRegEx::~PVXmlParamWidgetBoardSplitterRegEx
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardSplitterRegEx::~PVXmlParamWidgetBoardSplitterRegEx() = default;

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::allocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::allocBoardFields()
{
	tabParam = new QTabWidget(this);

	// tab
	name = new PVXmlParamWidgetEditorBox(QString("name"), new QVariant(node->attribute("name")));

	// tab regexp
	exp = new PVXmlParamTextEdit(QString("regexp"),
	                             QVariant(node->getDom().attribute("regexp", ".*")));
	labelNbr = new QLabel("");

	checkSaveValidLog = new QCheckBox("Save log sample in format file", this);

	QString textVal;
	for (auto & i : _data) {
		textVal += i;
		textVal += QChar('\n');
	}

	validWidget = new PVXmlParamTextEdit(QString("validator"), QVariant(textVal));
	table = new QTableWidget();
	btnApply = new QPushButton("Apply");
}

/******************************************************************************
 *
 * bool PVInspector::PVXmlParamWidgetBoardSplitterRegEx::confirmAndSave
 *
 *****************************************************************************/
bool PVInspector::PVXmlParamWidgetBoardSplitterRegEx::confirmAndSave()
{
	// open the confirm box.
	QDialog confirm(this);
	QVBoxLayout vb;
	confirm.setLayout(&vb);
	vb.addWidget(new QLabel("Do you want to apply the modifications ?"));
	QHBoxLayout bas;
	vb.addLayout(&bas);
	QPushButton no("No");
	bas.addWidget(&no);
	QPushButton yes("Yes");
	bas.addWidget(&yes);

	// connect the response button
	connect(&no, &QAbstractButton::clicked, &confirm, &QDialog::reject);
	connect(&yes, &QAbstractButton::clicked, &confirm, &QDialog::accept);

	// if confirmed then apply
	return confirm.exec();
}

/******************************************************************************
 *
 * VInspector::PVXmlParamWidgetBoardSplitterRegEx::createTab
 *
 *****************************************************************************/
QVBoxLayout* PVInspector::PVXmlParamWidgetBoardSplitterRegEx::createTab(const QString& title,
                                                                        QTabWidget* tab)
{
	auto tabWidget = new QWidget(tab);
	// create the layout
	auto tabWidgetLayout = new QVBoxLayout(tabWidget);

	// creation of the tab
	tabWidgetLayout->setContentsMargins(0, 0, 0, 0);
	tabWidget->setLayout(tabWidgetLayout);

	// add the tab
	tab->addTab(tabWidget, title);

	// return the layout to add items
	return tabWidgetLayout;
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::disableConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::disableConnexion()
{
	disconnect(name, &QLineEdit::textChanged, this,
	           &PVXmlParamWidgetBoardSplitterRegEx::slotSetValues);
	disconnect(exp, SIGNAL(textChanged(const QString&)), validWidget,
	           SLOT(setRegEx(const QString&)));
	disconnect(exp, &QTextEdit::textChanged, this,
	           &PVXmlParamWidgetBoardSplitterRegEx::slotSetValues);
	disconnect(exp, SIGNAL(textChanged(const QString&)), this, SLOT(regExCount(const QString&)));
	disconnect(validWidget, &QTextEdit::textChanged, this,
	           &PVXmlParamWidgetBoardSplitterRegEx::slotSetValues);
	disconnect(validWidget, &QTextEdit::textChanged, this,
	           &PVXmlParamWidgetBoardSplitterRegEx::slotNoteConfirmationNeeded);
	disconnect(btnApply, &QAbstractButton::clicked, this,
	           &PVXmlParamWidgetBoardSplitterRegEx::slotSetConfirmedValues);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::disAllocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::disAllocBoardFields()
{
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::draw
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::draw()
{
	// init layout
	auto qv = new QVBoxLayout();
	QVBoxLayout* tabReg = createTab("regexp", tabParam);
	QVBoxLayout* tabGeneral = createTab("general", tabParam);

	// init the parameter board layout
	qv->addWidget(tabParam);

	// tab general
	// field name
	tabGeneral->addWidget(new QLabel("RegEx name"));
	tabGeneral->addWidget(name);
	tabGeneral->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));
	// field expression

	// tab regexp
	tabReg->addWidget(new QLabel("Expression"));
	tabReg->addWidget(exp);
	// exp->setFocus();
	tabReg->addWidget(labelNbr);
	tabReg->addWidget(checkSaveValidLog);
	tabReg->addWidget(validWidget);
	tabReg->addWidget(new QLabel("view of selection"));
	tabReg->addWidget(table);
	// apply
	tabReg->addWidget(btnApply);
	// btnApply->setShortcut(QKeySequence(Qt::Key_Enter|Qt::Key_Return));
	btnApply->setShortcut(QKeySequence(Qt::Key_Return));

	setLayout(qv);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::exit
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::exit()
{
	PVLOG_DEBUG("slot PVInspector::PVXmlParamWidgetBoardSplitterRegEx::exit()\n");
	// open the confirmbox if we quit a regexp
	if (flagNeedConfirmAndSave && flagAskConfirmActivated) {
		if (confirmAndSave()) {
			slotSetConfirmedValues();
		}
		flagNeedConfirmAndSave = false;
		flagAskConfirmActivated = false;
	}

	disableConnexion();
	if (table->isVisible())
		table->setVisible(false);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::initConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::initConnexion()
{
	// tab general
	connect(name, &QLineEdit::textChanged, this,
	        &PVXmlParamWidgetBoardSplitterRegEx::slotSetValues); // to update tree view.
	connect(name, &QLineEdit::textChanged, this,
	        &PVXmlParamWidgetBoardSplitterRegEx::slotVerifRegExpInName); // to verify if regexp is
	                                                                     // write in name.
	// tab regexp
	connect(exp, SIGNAL(textChanged(const QString&)), validWidget,
	        SLOT(setRegEx(const QString&))); // to update regexp
	connect(exp, &QTextEdit::textChanged, this,
	        &PVXmlParamWidgetBoardSplitterRegEx::slotNoteConfirmationNeeded); // to note that we
	                                                                          // need to confirm
	                                                                          // change
	connect(exp, SIGNAL(textChanged(const QString&)), this,
	        SLOT(regExCount(const QString&))); // to update the numbre of field which are detected
	// connect(validWidget, SIGNAL(textChanged()), this, SLOT(slotNoteConfirmationNeeded()));//to
	// note that we need to confirm change
	connect(validWidget, &QTextEdit::textChanged, this,
	        &PVXmlParamWidgetBoardSplitterRegEx::slotUpdateTable); // to update the text validator
	connect(checkSaveValidLog, &QAbstractButton::clicked, this,
	        &PVXmlParamWidgetBoardSplitterRegEx::slotSaveValidator);
	connect(checkSaveValidLog, &QAbstractButton::clicked, this,
	        &PVXmlParamWidgetBoardSplitterRegEx::slotSetConfirmedValues);
	connect(btnApply, &QAbstractButton::clicked, this,
	        &PVXmlParamWidgetBoardSplitterRegEx::slotSetConfirmedValues); // to apply modification.
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::initValue
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::initValue()
{
	// init the number of field detected with the regexp
	regExCount(exp->toPlainText());
	// check or not the check box
	if (node->attribute("saveValidator", "").compare(QString("true")) == 0) {
		flagSaveRegExpValidator = true;
		checkSaveValidLog->setCheckState(Qt::Checked);
		validWidget->setVal(node->attribute("validator", true));
	} else {
		flagSaveRegExpValidator = false;
		checkSaveValidLog->setCheckState(Qt::Unchecked);
		validWidget->setVal(node->attribute("validator", false));
	}
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::getWidgetToFocus
 *
 *****************************************************************************/
QWidget* PVInspector::PVXmlParamWidgetBoardSplitterRegEx::getWidgetToFocus()
{
	return (QWidget*)name;
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::regExCount
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::regExCount(const QString& reg)
{
	// set regexp
	QRegExp regExp = QRegExp(reg);
	nbr = regExp.captureCount();
	// set tne number of field
	labelNbr->setText(QString("selection count : %1").arg(nbr));

	if (nbr >= 1) { // if there is at least one selection...
		// get selection detection pettern.
		QString patternToDetectSel;
		for (int i = 0; i < nbr; i++) {
			patternToDetectSel += QString(".*(\\([^)]*\\)).*");
		}
		QRegExp regDetectSel = QRegExp(QString(patternToDetectSel));
		// detect selections in regexp.
		regDetectSel.indexIn(reg, 0);
		// set string list of each selection pattern
		for (int i = 1; i <= nbr; i++) { // for each selection...
			node->setAttribute(QString("selectionRegExp-%0").arg(i),
			                   regDetectSel.cap(i).toUtf8().constData());
		}
	}
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotNoteConfirmationNeeded
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotNoteConfirmationNeeded()
{
	PVLOG_DEBUG(
	    "slot PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotNoteConfirmationNeeded()\n");
	flagNeedConfirmAndSave = true; // note that we need confirmation
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotOpenLogValid
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotOpenLogValid()
{
	QString urlFile = PVWidgets::PVFileDialog::getOpenFileName(nullptr, QString("Select the file."),
	                                                           "."); // open file chooser
	QFile f(urlFile);
	if (f.exists()) { // if the file is valid
		if (!f.open(QIODevice::ReadOnly | QIODevice::Text))
			return;
		for (int i = 0; i < 50; i++) { // for the 50 first line ...
			QString l = validWidget->getVal().toString();
			l.push_back(QString(f.readLine())); //...add the line in validator
			validWidget->setVal(l);
		}
	}
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSaveValidator
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSaveValidator(bool stat)
{
	flagSaveRegExpValidator = stat;
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSetConfirmedValues
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSetConfirmedValues()
{
	slotSetValues();                                           // save various value
	node->setAttribute(QString("regexp"), exp->toPlainText()); // save expression
	node->setAttribute(QString("validator"), validWidget->getVal().toString(),
	                   flagSaveRegExpValidator); // save the text in validator

	regExCount(exp->toPlainText());
	node->setNbr(nbr); // set the fileds with expression rexexp selection count.
	flagNeedConfirmAndSave = false;
	Q_EMIT signalRefreshView();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSetValues
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotSetValues()
{
	node->setAttribute(QString("name"), name->text()); // save name
	if (flagSaveRegExpValidator) {
		node->setAttribute(QString("saveValidator"), "true");
	} else {
		node->setAttribute(QString("saveValidator"), "false");
	}

	Q_EMIT signalRefreshView();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotShowTable
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotShowTable(bool isVisible)
{
	table->setVisible(isVisible); // hide or show table validator selection
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotVerifRegExpInName
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotVerifRegExpInName()
{
	PVLOG_DEBUG("PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotVerifRegExpInName\n");
	// char we want to detecte in the name
	QRegExp reg(R"(.*(\*|\[|\{|\]|\}).*)");
	if (reg.exactMatch(name->text())) { // if there is
		// create the confirm popup
		QDialog confirm(this);
		QVBoxLayout vb;
		confirm.setLayout(&vb);
		vb.addWidget(new QLabel("you are maybe writting a regexp in the name field. Do you want to "
		                        "write it in the expression field ?"));
		QHBoxLayout bas;
		vb.addLayout(&bas);
		QPushButton no("No");
		bas.addWidget(&no);
		QPushButton yes("Yes");
		bas.addWidget(&yes);

		// connect the response button
		connect(&no, &QAbstractButton::clicked, &confirm, &QDialog::reject);
		connect(&yes, &QAbstractButton::clicked, &confirm, &QDialog::accept);

		if (confirm.exec()) {           // if confirmed then apply...
			exp->setText(name->text()); // push text
			name->setText("");
		}
	}
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotUpdateTable
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::slotUpdateTable()
{
	QRegExp reg = QRegExp(exp->toPlainText());

	// update the number of column
	reg.indexIn(validWidget->getVal().toString(), 0);
	table->setColumnCount(reg.captureCount());

	// feed each line with the matching in text zone.
	QStringList myText = validWidget->getVal().toString().split("\n");
	table->setRowCount(myText.count());
	updateHeaderTable();
	for (int line = 0; line < myText.count(); line++) { // for each line...
		QString myLine = myText.at(line);
		if (reg.indexIn(myLine, 0)) {
			for (int cap = 0; cap < reg.captureCount();
			     cap++) { // for each column (regexp selection)...
				reg.indexIn(myLine, 0);
				table->setItem(line, cap, new QTableWidgetItem(reg.cap(cap + 1)));
				int width = 12 + (8 * reg.cap(cap + 1).length());
				if (width > table->columnWidth(cap)) {
					table->setColumnWidth(cap, width); // update the size
				}
			}
		}
	}
	table->setContentsMargins(3, 0, 3, 0);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::updateHeaderTable
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardSplitterRegEx::updateHeaderTable()
{
	QStringList l;
	for (int i = 0; i < node->countChildren(); i++) {
		l.push_back(node->getChild(i)->getOutName());
		table->setColumnWidth(i, 1);
		int width = 12 + (8 * node->getChild(i)->getOutName().length());
		if (width > table->columnWidth(i)) {
			table->setColumnWidth(i, width); // update the size
		}
	}
	// qDebug()<<l;
	// table->setHorizontalHeaderItem(1,"e")
	table->setHorizontalHeaderLabels(l);
	table->setContentsMargins(3, 0, 3, 0);
}
