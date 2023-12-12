/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVXMLPARAMTEXTEDIT_H
#define PVXMLPARAMTEXTEDIT_H
#include <QTextEdit>
#include <QString>
#include <QVariant>
#include <QDebug>
#include <QVariant>
#include <QSyntaxHighlighter>
#include <QRegExp>

namespace App
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
} // namespace App
#endif /* PVXMLPARAMTEXTEDIT_H */
