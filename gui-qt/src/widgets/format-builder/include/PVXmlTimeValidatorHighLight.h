//! \file PVXmlTimeValidatorHighLight.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVXMLTIMEVALIDATORHIGHLIGHT_H
#define	PVXMLTIMEVALIDATORHIGHLIGHT_H

#include <QSyntaxHighlighter>
#include <QDebug>
#include <QString>
#include <QStringList>
#include <QTextEdit>
#include <QChar>
#include <QColor>
#include <QRegExp>
#include <QDateTime>

#include <PVXmlParamTextEdit.h>



//class QTextEdit;
namespace PVInspector{
  
class PVXmlTimeValidatorHighLight : public QSyntaxHighlighter {
     Q_OBJECT
     
public:
    PVXmlTimeValidatorHighLight(PVXmlParamTextEdit* pParent, const QStringList &myDateFormats);
    /**
     * Modify de regular expression.
     * @param pRegStr
     */
    void setDateFormat(const QStringList &pFormatStr);
    virtual ~PVXmlTimeValidatorHighLight();
    /**
     * Function subclass to define highlighting rules.
     * @param 
     */
    virtual void highlightBlock(const QString&);
    
private:
        PVXmlParamTextEdit *aParent;
        QStringList formatStr;
};

}

#endif	/* PVXMLREGVALIDATORHIGHLIGHT_H */
