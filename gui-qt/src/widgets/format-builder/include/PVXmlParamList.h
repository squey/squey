/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVXMLPARAMLIST_H
#define PVXMLPARAMLIST_H

#include <QListWidget>
#include <QString>
#include <QStringList>

namespace PVInspector
{

class PVXmlParamList : public QListWidget
{
	Q_OBJECT
  public:
	PVXmlParamList(QString name);

  public:
	void setItems(QStringList const& l);
	QStringList selectedList();
	void select(QStringList const& l);
	QString const& name() const { return _name; }

  protected:
	QString _name;
};
} // namespace PVInspector

#endif
