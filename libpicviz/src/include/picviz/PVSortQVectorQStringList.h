//! \file PVSortQVectorQStringList.h
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011



#ifndef PICVIZ_PVSORTVECTOR_H
#define PICVIZ_PVSORTVECTOR_H

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
#include <QObject>

#include <pvcore/general.h>
#include <picviz/PVSortQVectorQStringListThread.h>
//#include <PVProgressBox.h>

#include <pvrush/PVNraw.h>

namespace Picviz {

// AG: this class should now be renamed.. (or implemented as a template class)
class PVLibPicvizDecl PVSortQVectorQStringList:public QObject/*:public QDialog*/{
Q_OBJECT
	

public:
	PVSortQVectorQStringList(QObject *parent=0);
	~PVSortQVectorQStringList();
	
	
	void setList(PVRush::PVNraw::nraw_table *, QVector<int>* );
	void sort(int idColumn, Qt::SortOrder order );
	PVSortQVectorQStringListThread *myThread;
	
private:
	//dialog for waitting during sort
	//PVInspector::PVProgressBox *dialogBox;
	void setDialog();

public slots:
	void closeDialog();
	

};
	
}

//typedef bool(*) LessThan(const QPair<QString*, int> &, const QPair<QString*, int> &);


#endif 
