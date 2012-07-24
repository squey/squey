/**
 * \file PVXmlRegValidatorHighLight.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVXMLREGVALIDATORHIGHLIGHT_H
#define	PVXMLREGVALIDATORHIGHLIGHT_H

#include <QSyntaxHighlighter>
#include <QDebug>
#include <QString>
#include <QTextEdit>
#include <QChar>
#include <QColor>
#include <QRegExp>

#include "PVXmlParamTextEdit.h"



//class QTextEdit;
namespace PVInspector{
class PVXmlRegValidatorHighLight : public QSyntaxHighlighter {
     Q_OBJECT
     
public:
    PVXmlRegValidatorHighLight(QTextEdit* pParent);
    /**
     * Modify de regular expression.
     * @param pRegStr
     */
    void setRegExp( const QString &pRegStr);
    virtual ~PVXmlRegValidatorHighLight();
    /**
     * Function subclass to define highlighting rules.
     * @param 
     */
    virtual void highlightBlock(const QString&);
    
private:
        PVXmlParamTextEdit *aParent;
        QString aRegExp;
};
}
#endif	/* PVXMLREGVALIDATORHIGHLIGHT_H */

