/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVREGEXPEDITOR_H
#define PVCORE_PVREGEXPEDITOR_H

#include <QLineEdit>
#include <QRegExp>

namespace PVWidgets
{

/**
 * \class PVRegexpEditor
 */
class PVRegexpEditor : public QLineEdit
{
	Q_OBJECT
	Q_PROPERTY(QRegExp _rx READ get_rx WRITE set_rx USER true)

  public:
	PVRegexpEditor(QWidget* parent = 0);
	virtual ~PVRegexpEditor();

  public:
	QRegExp get_rx() const;
	void set_rx(QRegExp rx);
};
}

#endif
