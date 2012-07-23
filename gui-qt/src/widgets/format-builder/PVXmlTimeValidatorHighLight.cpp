/**
 * \file PVXmlTimeValidatorHighLight.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <PVXmlTimeValidatorHighLight.h>
#include <PVXmlParamTextEdit.h>

#include <pvkernel/core/PVDateTimeParser.h>
#include <unicode/calendar.h>
#include <unicode/unistr.h>


/******************************************************************************
 *
 * PVInspector::PVXmlTimeValidatorHighLight::PVXmlTimeValidatorHighLight
 *
 *****************************************************************************/
PVInspector::PVXmlTimeValidatorHighLight::PVXmlTimeValidatorHighLight(PVXmlParamTextEdit* pParent, const QStringList &myDateFormats) :
    QSyntaxHighlighter(pParent->document()) {
    //qDebug()<<"PVXmlTimeValidatorHighLight::PVXmlTimeValidatorHighLight("<<pParent->getVal().toString()<<","<<myDateFormats<<")";
    setDocument(pParent->document());
  
     
    aParent = (PVXmlParamTextEdit*) pParent;
    setObjectName("valide");

    
}



/******************************************************************************
 *
 * PVInspector::PVXmlTimeValidatorHighLight::setDateFormat
 *
 *****************************************************************************/
void PVInspector::PVXmlTimeValidatorHighLight::setDateFormat(const QStringList &pFormatStr) {
    formatStr = pFormatStr;
    rehighlight();
}



/******************************************************************************
 *
 * PVInspector::PVXmlTimeValidatorHighLight::~PVXmlTimeValidatorHighLight
 *
 *****************************************************************************/
PVInspector::PVXmlTimeValidatorHighLight::~PVXmlTimeValidatorHighLight() {
}




/******************************************************************************
 *
 * PVInspector::PVXmlTimeValidatorHighLight::highlightBlock
 *
 *****************************************************************************/
void PVInspector::PVXmlTimeValidatorHighLight::highlightBlock(const QString &text) {
 

    //define format
    QTextCharFormat formatMacthLine;
    QTextCharFormat formatMacthLineSelection;
    QTextCharFormat formatNoMatch;
    formatMacthLine.setForeground(Qt::black);
    formatMacthLineSelection.setForeground(Qt::black);
    formatMacthLineSelection.setBackground(QColor("#c0c0c0"));
    formatNoMatch.setForeground(QColor("#FF0000"));
    
    //default color
    setFormat(0, text.count(), formatNoMatch);
    
	// Check if this is parseable
	PVCore::PVDateTimeParser parser(formatStr);
	UErrorCode err = U_ZERO_ERROR;
	Calendar* cal = Calendar::createInstance(err);
	if (parser.mapping_time_to_cal(text, cal)) {
		setFormat(0, text.count(), formatMacthLine);
	}
	delete cal;
}
