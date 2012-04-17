#ifndef __PVAD2GEDGEEDITOR_H__
#define __PVAD2GEDGEEDITOR_H__

#include <QtGui>

namespace Picviz {

class PVAD2GEdgeEditor : public QDialog
{
	Q_OBJECT
public:
	PVAD2GEdgeEditor(QWidget* parent = 0);

public slots:
	void add_function_Slot();
	void edit_function_Slot();
	void remove_function_Slot();
};

class PVAD2GFunctionProperties : public QDialog
{
	Q_OBJECT
public:
	PVAD2GFunctionProperties(QWidget* parent = 0);
};

}

#endif // __PVAD2GEDGEEDITOR_H__
