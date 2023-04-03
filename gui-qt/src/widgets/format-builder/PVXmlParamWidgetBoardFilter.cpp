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

#include <PVXmlParamWidgetBoardFilter.h>

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::PVXmlParamWidgetBoardFilter
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardFilter::PVXmlParamWidgetBoardFilter(
    PVRush::PVXmlTreeNodeDom* pNode, PVXmlParamWidget* parent)
    : QWidget(), _parent(parent)
{
	setObjectName("PVXmlParamWidgetBoardFilter");
	node = pNode;
	allocBoardFields();
	draw();
	initValue();
	initConnexion();
	validWidget->setRegEx(exp->toPlainText());
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::~PVXmlParamWidgetBoardFilter
 *
 *****************************************************************************/
PVInspector::PVXmlParamWidgetBoardFilter::~PVXmlParamWidgetBoardFilter()
{
	disableConnexion();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::allocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::allocBoardFields()
{
	// allocate each field and init them with the saved value
	name = new PVXmlParamWidgetEditorBox(QString("name"), new QVariant(node->attribute("name")));
	exp = new PVXmlParamTextEdit(QString("regexp"),
	                             QVariant(node->getDom().attribute("regexp", ".*")));
	validWidget =
	    new PVXmlParamTextEdit(QString("validator"), QVariant(node->attribute("validator")));
	typeOfFilter = new PVXmlParamComboBox("reverse");
	typeOfFilter->addItem("include");
	typeOfFilter->addItem("exclude");
	typeOfFilter->select((node->attribute("reverse") == "true") ? "exclude" : "include");
	buttonNext = new QPushButton("Next");
	buttonNext->setShortcut(QKeySequence(Qt::Key_Return));
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::disableConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::disableConnexion()
{
	disconnect(name, &QLineEdit::textChanged, this, &PVXmlParamWidgetBoardFilter::slotSetValues);
	disconnect(exp, SIGNAL(textChanged()), validWidget, SLOT(setRegEx(const QString&)));
	disconnect(exp, &QTextEdit::textChanged, this, &PVXmlParamWidgetBoardFilter::slotSetValues);
	disconnect(validWidget, &QTextEdit::textChanged, this,
	           &PVXmlParamWidgetBoardFilter::slotSetValues);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::disAllocBoardFields
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::disAllocBoardFields()
{
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::draw
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::draw()
{
	auto qv = new QVBoxLayout();

	qv->addWidget(new QLabel("Filter name"));
	qv->addWidget(name);
	qv->addWidget(typeOfFilter);
	qv->addWidget(new QLabel("Expression"));
	qv->addWidget(exp);
	qv->addWidget(new QLabel("Expression validator"));
	qv->addWidget(validWidget);
	qv->addWidget(buttonNext);

	qv->setStretchFactor(exp, 4);
	qv->setStretchFactor(validWidget, 2);

	setLayout(qv);

	// name->textCursor().movePosition(QTextCursor::Start);
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::getWidgetToFocus
 *
 *****************************************************************************/
QWidget* PVInspector::PVXmlParamWidgetBoardFilter::getWidgetToFocus()
{
	return (QWidget*)name;
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::initConnexion
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::initConnexion()
{
	connect(name, &QLineEdit::textChanged, this, &PVXmlParamWidgetBoardFilter::slotSetValues);
	connect(name, &QLineEdit::textChanged, this,
	        &PVXmlParamWidgetBoardFilter::slotVerifRegExpInName);
	connect(exp, SIGNAL(textChanged()), validWidget, SLOT(setRegEx()));
	connect(exp, &QTextEdit::textChanged, this, &PVXmlParamWidgetBoardFilter::slotSetValues);
	connect(validWidget, &QTextEdit::textChanged, this,
	        &PVXmlParamWidgetBoardFilter::slotSetValues);
	connect(typeOfFilter, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(slotSetValues()));
	connect(buttonNext, &QAbstractButton::clicked, this,
	        &PVXmlParamWidgetBoardFilter::slotEmitNext);
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::initValue
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::initValue()
{
}

/******************************************************************************
 *
 * PVInspector::PVXmlParamWidgetBoardFilter::slotEmitNext
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::slotEmitNext()
{
	Q_EMIT signalEmitNext();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::slotSetValues
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::slotSetValues()
{ // called when we modifi something.
	node->setAttribute(QString("name"), name->text());
	node->setAttribute(QString("regexp"), exp->toPlainText());
	node->setAttribute(QString("validator"), validWidget->getVal().toString());
	node->setAttribute(QString("reverse"),
	                   (typeOfFilter->val().toString() == "exclude") ? "true" : "false");

	Q_EMIT signalRefreshView();
}

/******************************************************************************
 *
 * void PVInspector::PVXmlParamWidgetBoardFilter::slotVerifRegExpInName
 *
 *****************************************************************************/
void PVInspector::PVXmlParamWidgetBoardFilter::slotVerifRegExpInName()
{
	// char we want to detecte in the name
	QRegExp reg(R"(.*(\*|\[|\{|\]|\}).*)");
	if (reg.exactMatch(name->text())) {
		// create and open the confirm box
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

		// if confirmed then apply
		if (confirm.exec()) {
			exp->setText(name->text());
			name->setText("");
		}
	}
}
