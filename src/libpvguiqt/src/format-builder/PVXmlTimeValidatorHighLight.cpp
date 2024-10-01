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

#include <PVXmlTimeValidatorHighLight.h>
#include <PVXmlParamTextEdit.h>

#include <pvkernel/core/PVDateTimeParser.h>
#include <unicode/calendar.h>
#include <unicode/unistr.h>

using namespace icu_75;

/******************************************************************************
 *
 * App::PVXmlTimeValidatorHighLight::PVXmlTimeValidatorHighLight
 *
 *****************************************************************************/
App::PVXmlTimeValidatorHighLight::PVXmlTimeValidatorHighLight(
    PVXmlParamTextEdit* pParent, const QStringList& /*myDateFormats*/)
    : QSyntaxHighlighter(pParent->document())
{
	// qDebug()<<"PVXmlTimeValidatorHighLight::PVXmlTimeValidatorHighLight("<<pParent->getVal().toString()<<","<<myDateFormats<<")";
	setDocument(pParent->document());

	aParent = (PVXmlParamTextEdit*)pParent;
	setObjectName("valide");
}

/******************************************************************************
 *
 * App::PVXmlTimeValidatorHighLight::setDateFormat
 *
 *****************************************************************************/
void App::PVXmlTimeValidatorHighLight::setDateFormat(const QStringList& pFormatStr)
{
	formatStr = pFormatStr;
	rehighlight();
}

/******************************************************************************
 *
 * App::PVXmlTimeValidatorHighLight::~PVXmlTimeValidatorHighLight
 *
 *****************************************************************************/
App::PVXmlTimeValidatorHighLight::~PVXmlTimeValidatorHighLight() = default;

/******************************************************************************
 *
 * App::PVXmlTimeValidatorHighLight::highlightBlock
 *
 *****************************************************************************/
void App::PVXmlTimeValidatorHighLight::highlightBlock(const QString& text)
{

	// define format
	QTextCharFormat formatMacthLine;
	QTextCharFormat formatMacthLineSelection;
	QTextCharFormat formatNoMatch;
	formatMacthLine.setForeground(Qt::black);
	formatMacthLineSelection.setForeground(Qt::black);
	formatMacthLineSelection.setBackground(QColor("#c0c0c0"));
	formatNoMatch.setForeground(QColor("#FF0000"));

	// default color
	setFormat(0, text.size(), formatNoMatch);

	// Check if this is parseable
	PVCore::PVDateTimeParser parser(formatStr);
	UErrorCode err = U_ZERO_ERROR;
	Calendar* cal = Calendar::createInstance(err);
	if (parser.mapping_time_to_cal(text, cal)) {
		setFormat(0, text.size(), formatMacthLine);
	}
	delete cal;
}
