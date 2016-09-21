/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVXMLPARAMTEXTEDIT_H
#define PVXMLPARAMTEXTEDIT_H
#include <QTextEdit>
#include <QString>
#include <QVariant>
#include <QDebug>
#include <QVariant>
#include <QSyntaxHighlighter>

namespace PVInspector
{

class PVXmlRegValidatorHighLight;
class PVXmlTimeValidatorHighLight;

class PVXmlParamTextEdit : public QTextEdit
{
	Q_OBJECT
  public:
	enum TypeOfTextEdit { text, regexpValid, dateValid };

	PVXmlParamTextEdit(QString pName, QVariant var);
	~PVXmlParamTextEdit() override;

	/**
	 * Get the new textContent.
	 * @return
	 */
	QVariant getVal() const;

	/**
	 * Setup the text content.
	 * @param
	 */
	void setVal(const QString&);

	void validDateFormat(const QStringList&);

	TypeOfTextEdit typeOfTextEdit;

  private:
	QString variable;
	QRegExp regExp;
	QStringList format;
	PVXmlRegValidatorHighLight* highlight;
	PVXmlTimeValidatorHighLight* timeValid;
	bool editing;

  public Q_SLOTS:
	/**
	 * Event when something modify the regular expression.
	 * @param regStr
	 */
	void setRegEx();
	void setRegEx(const QString& regEx);
	void slotHighLight();
};
} // namespace PVInspector
#endif /* PVXMLPARAMTEXTEDIT_H */
