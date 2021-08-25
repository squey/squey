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

#include <PVXmlRegValidatorHighLight.h>
#include <PVXmlParamTextEdit.h>

/******************************************************************************
 *
 * PVInspector::PVXmlRegValidatorHighLight::PVXmlRegValidatorHighLight
 *
 *****************************************************************************/
PVInspector::PVXmlRegValidatorHighLight::PVXmlRegValidatorHighLight(QTextEdit* pParent)
    : QSyntaxHighlighter(pParent->document())
{
	setDocument(pParent->document());
	aParent = (PVXmlParamTextEdit*)pParent;
	setObjectName("valide");
}

/******************************************************************************
 *
 * void PVInspector::PVXmlRegValidatorHighLight::setRegExp
 *
 *****************************************************************************/
void PVInspector::PVXmlRegValidatorHighLight::setRegExp(const QString& pRegStr)
{
	aRegExp = pRegStr;
	rehighlight();
}

/******************************************************************************
 *
 * PVInspector::PVXmlRegValidatorHighLight::~PVXmlRegValidatorHighLight
 *
 *****************************************************************************/
PVInspector::PVXmlRegValidatorHighLight::~PVXmlRegValidatorHighLight() = default;

/******************************************************************************
 *
 * void PVInspector::PVXmlRegValidatorHighLight::highlightBlock
 *
 *****************************************************************************/
void PVInspector::PVXmlRegValidatorHighLight::highlightBlock(const QString& text)
{

	// define format
	QTextCharFormat formatMacthLine;
	QTextCharFormat formatMacthLineSelection;
	QTextCharFormat formatNoMatch;
	formatMacthLine.setForeground(Qt::black);
	formatMacthLineSelection.setForeground(Qt::black);
	formatMacthLineSelection.setBackground(QColor("#c0c0c0"));
	formatNoMatch.setForeground(QColor("#707070"));

	QRegExp regExp(aRegExp); // init regexp

	if (regExp.exactMatch(text)) {                   // if line is matching
		setFormat(0, text.count(), formatMacthLine); // set text color black
		regExp.indexIn(text, 0);

		// selection coloring
		for (int i = 1; i <= regExp.matchedLength(); i++) {
			setFormat(regExp.pos(i), regExp.cap(i).length(),
			          formatMacthLineSelection); // underline each selection
		}

	} else {                                       // if line is not matching
		setFormat(0, text.count(), formatNoMatch); // set text color gray
	}
}
