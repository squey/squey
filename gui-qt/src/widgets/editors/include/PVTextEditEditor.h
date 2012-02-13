#ifndef PVCORE_PVTEXTEDITEDITOR_H
#define PVCORE_PVTEXTEDITEDITOR_H

#include <QWidget>
#include <QTextEdit>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVTextEditType.h>

#include <picviz/PVView.h>

namespace PVInspector {

/**
 * \class PVTextEditEditor
 */
class PVTextEditEditor: public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVTextEditType _text READ get_text WRITE set_text USER true)

public:
	PVTextEditEditor(Picviz::PVView& view, QWidget *parent = 0);

public:
	PVCore::PVTextEditType get_text() const;
	void set_text(PVCore::PVTextEditType const& text);

protected:
	virtual bool eventFilter(QObject *object, QEvent *event);

protected slots:
	void slot_import_file();

protected:
	QTextEdit* _text_edit;
	Picviz::PVView& _view;
};

}

#endif
