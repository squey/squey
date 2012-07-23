/**
 * \file PVXmlRegValidatorHighLight.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <PVXmlRegValidatorHighLight.h>
#include <PVXmlParamTextEdit.h>




/******************************************************************************
 *
 * PVInspector::PVXmlRegValidatorHighLight::PVXmlRegValidatorHighLight
 *
 *****************************************************************************/
PVInspector::PVXmlRegValidatorHighLight::PVXmlRegValidatorHighLight(QTextEdit* pParent) :
    QSyntaxHighlighter(pParent->document()) {
    setDocument(pParent->document());
    aParent = (PVXmlParamTextEdit*) pParent;
    setObjectName("valide");
}



/******************************************************************************
 *
 * void PVInspector::PVXmlRegValidatorHighLight::setRegExp
 *
 *****************************************************************************/
void PVInspector::PVXmlRegValidatorHighLight::setRegExp(const QString &pRegStr) {
    aRegExp = pRegStr;
    rehighlight();
}



/******************************************************************************
 *
 * PVInspector::PVXmlRegValidatorHighLight::~PVXmlRegValidatorHighLight
 *
 *****************************************************************************/
PVInspector::PVXmlRegValidatorHighLight::~PVXmlRegValidatorHighLight() {
}



/******************************************************************************
 *
 * void PVInspector::PVXmlRegValidatorHighLight::highlightBlock
 *
 *****************************************************************************/
void PVInspector::PVXmlRegValidatorHighLight::highlightBlock(const QString &text) {
 
    //define format
    QTextCharFormat formatMacthLine;
    QTextCharFormat formatMacthLineSelection;
    QTextCharFormat formatNoMatch;
    formatMacthLine.setForeground(Qt::black);
    formatMacthLineSelection.setForeground(Qt::black);
    formatMacthLineSelection.setBackground(QColor("#c0c0c0"));
    formatNoMatch.setForeground(QColor("#707070"));



    QRegExp regExp(aRegExp);//init regexp
    
    if (regExp.exactMatch(text)) {//if line is matching
        setFormat(0, text.count(), formatMacthLine);//set text color black
        regExp.indexIn(text, 0);

        //selection coloring
        for (int i = 1; i <= regExp.matchedLength(); i++) {
            setFormat(regExp.pos(i), regExp.cap(i).length(), formatMacthLineSelection);//underline each selection
        }

    } else {//if line is not matching
        setFormat(0, text.count(), formatNoMatch);//set text color gray
    }
}