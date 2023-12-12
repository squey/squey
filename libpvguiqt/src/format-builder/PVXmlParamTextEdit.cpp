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

#include <PVXmlParamTextEdit.h>
#include <PVXmlRegValidatorHighLight.h>
#include <PVXmlTimeValidatorHighLight.h>

#include <QSizePolicy>

/******************************************************************************
 *
 * App::PVXmlParamTextEdit::PVXmlParamTextEdit
 *
 *****************************************************************************/
App::PVXmlParamTextEdit::PVXmlParamTextEdit(QString pName, QVariant var) : QTextEdit()
{
	setObjectName(pName);
	variable = var.toString();
	setText(variable);

	typeOfTextEdit = text;
	editing = true;

	QSizePolicy sp(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);
	sp.setHeightForWidth(sizePolicy().hasHeightForWidth());
	setSizePolicy(sp);

	connect(this, &QTextEdit::textChanged, this, &PVXmlParamTextEdit::slotHighLight);
}

/******************************************************************************
 *
 * App::PVXmlParamTextEdit::~PVXmlParamTextEdit
 *
 *****************************************************************************/
App::PVXmlParamTextEdit::~PVXmlParamTextEdit() = default;

/******************************************************************************
 *
 * App::PVXmlParamTextEdit::setRegEx
 *
 *****************************************************************************/
void App::PVXmlParamTextEdit::setRegEx()
{
	setRegEx(((PVXmlParamTextEdit*)sender())->toPlainText());
}

void App::PVXmlParamTextEdit::setRegEx(const QString& regExp)
{
	if (editing) {
		highlight = new PVXmlRegValidatorHighLight(
		    (PVXmlParamTextEdit*)this); // we are in regexp validator case
		editing = false;
	}
	typeOfTextEdit = regexpValid; // define type as a regexp validator
	highlight->setRegExp(regExp);
	highlight->rehighlight();
}

/******************************************************************************
 *
 * App::PVXmlParamTextEdit::slotHighLight
 *
 *****************************************************************************/
void App::PVXmlParamTextEdit::slotHighLight()
{
	if (typeOfTextEdit == dateValid) {
		// qDebug()<<"PVXmlParamTextEdit::slotHighLight for axis";
		// timeValid->rehighlight();
	}
}

/******************************************************************************
 *
 * QVariant App::PVXmlParamTextEdit::getVal
 *
 *****************************************************************************/
QVariant App::PVXmlParamTextEdit::getVal() const
{
	// variable=QString(toPlainText());
	return QVariant(toPlainText());
}

/******************************************************************************
 *
 * App::PVXmlParamTextEdit::setVal
 *
 *****************************************************************************/
void App::PVXmlParamTextEdit::setVal(const QString& val)
{
	variable = val;
	setText(variable);
}

/******************************************************************************
 *
 * App::PVXmlParamTextEdit::validDateFormat
 *
 *****************************************************************************/
void App::PVXmlParamTextEdit::validDateFormat(const QStringList& pFormat)
{
	if (editing) {
		timeValid = new PVXmlTimeValidatorHighLight(this, format); // we are in time validator case
		editing = false;
	}
	typeOfTextEdit = dateValid; // define type as a date validator.
	format = pFormat;
	timeValid->setDateFormat(format); // set the format to validate.
	timeValid->rehighlight();         // update color for validated line
}
