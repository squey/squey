//! \file PVSortQVectorQStringListThread.h
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011



#ifndef PICVIZ_PVSORTVECTORTHREAD_H
#define PICVIZ_PVSORTVECTORTHREAD_H

//import qt
#include <QVector>
#include <QString>
#include <QStringList>
#include <QPair>
#include <QThread>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>

#include <pvcore/general.h>
#include <pvrush/PVNraw.h>

#ifdef WIN32
#ifdef picviz_EXPORTS
#define PVLibExport __declspec(dllexport)
#else
#define PVLibExport __declspec(dllimport)
#endif
#else
#define PVLibExport
#endif

namespace Picviz {

class PVLibExport PVSortQVectorQStringListThread: public QThread{
Q_OBJECT	

public:
	PVSortQVectorQStringListThread(QObject *parent=0);
	
	void setList(PVRush::PVNraw::nraw_table *, QVector<int>* );
	void sort(int idColumn, Qt::SortOrder order );
	
	void swap(int i, int j);
	int compare(int i, int j, int col);
	
	//comparison functions
	static bool itemLessThan(const QPair<QString*, int> &left, const QPair<QString*, int> &right);
	static bool itemGreaterThan(const QPair<QString*, int> &left, const QPair<QString*, int> &right);
	
	void run();
	void init(int idColumn, Qt::SortOrder p_order );
	void update();
	void setDialog(QDialog*);
	

	
//private:
	//data to sort
	PVRush::PVNraw::nraw_table *myTable;
	QVector<int> *correspondTable;
	
	//sort parameters
	QVector<QPair<QString*, int> > sortable;
	Qt::SortOrder order;
	int sizeData;
	
	QDialog *diag;
/*	
signals:
	void theend();*/
	
};
	
}

//typedef bool(*) LessThan(const QPair<QString*, int> &, const QPair<QString*, int> &);


#endif 
