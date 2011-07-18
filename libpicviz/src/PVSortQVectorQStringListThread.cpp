//! \file PVSortQVectorQStringListThread.cpp
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVSortQVectorQStringListThread.h>
#include <QFuture>
#include <QtCore>



/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::PVSortQVectorQStringListThread
 *
 *****************************************************************************/
Picviz::PVSortQVectorQStringListThread::PVSortQVectorQStringListThread(QObject *parent):QThread(parent){
	correspondTable=0;myTable=0;
}



/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::run
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringListThread::run(){
	///TODO use toLocal8bits (and strcoll in compare function) to save time.
	PVLOG_INFO("start thread\n");
	//choose order
	bool (*compare) (const QPair<QString*, int> &, const QPair<QString*, int> &);
	compare = (order == Qt::AscendingOrder ? &itemLessThan : &itemGreaterThan);
	//sort processing
	qStableSort(sortable.begin(), sortable.end(), compare);///TODO test
	
	//emit theend();
	/*PVLOG_INFO("trying to close cancel\n");
	diag->accept();*/
	PVLOG_INFO("end   thread\n");
}

/******************************************************************************
 *
 * extrun
 *	NOT USED!
 *
 *****************************************************************************/
void extrun(Picviz::PVSortQVectorQStringListThread *c){
	//choose order
	bool (*compare) (const QPair<QString*, int> &, const QPair<QString*, int> &);
	compare = (c->order == Qt::AscendingOrder ? &c->itemLessThan : &c->itemGreaterThan);
	//sort processing
	qStableSort(c->sortable.begin(), c->sortable.end(), compare);///TODO test
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::setDialog
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringListThread::setDialog(QDialog *d){
	diag = d;
}



/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::setList
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringListThread::setList(PVRush::PVNraw::nraw_table *dataIn, std::vector<int>* correspondingTable){
	myTable = dataIn;
	correspondTable=correspondingTable;
}


/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::sortByColumn
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringListThread::sort(int , Qt::SortOrder  ){
	PVLOG_WARN("Picviz::PVSortQVectorQStringListThread::sort(...) is not implemented");
	/*PVLOG_INFO("start   thread\n");
	QFuture<void> f = QtConcurrent::run(extrun,this);
	f.waitForFinished();
	emit finished();
	PVLOG_INFO("end   thread\n");*/
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::swap
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringListThread::swap(int i, int j){
	
	/*********************************
	**   update corresponding list	**
	*********************************/
	int old_second=correspondTable->at(j);
	correspondTable->at(j)=correspondTable->at(i);
        correspondTable->at(i)=old_second;
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::compare
 *
 *****************************************************************************/
int Picviz::PVSortQVectorQStringListThread::compare(int i, int j, int col){
	PVRush::PVNraw::nraw_table_line l1 = myTable->at(i);//get first line
	QString item1 = l1.at(col);//get fisrt item
	PVRush::PVNraw::nraw_table_line l2 = myTable->at(j);//get second line
	QString item2 = l2.at(col);//get second item
	//return the item string compare result
	return item1.compare(item2);
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::itemLessThan
 *
 *****************************************************************************/
bool Picviz::PVSortQVectorQStringListThread::itemLessThan(const QPair<QString*, int> &left, const QPair<QString*, int> &right){
	//string comparison
	int cmp = left.first->compare(right.first);
	return cmp<0;
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::itemGreaterThan
 *
 *****************************************************************************/
bool Picviz::PVSortQVectorQStringListThread::itemGreaterThan(const QPair<QString*, int> &left, const QPair<QString*, int> &right){
	//string comparison
	int cmp = left.first->compare(right.first);
	return cmp>0;
}
/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::init
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringListThread::init(int idColumn, Qt::SortOrder p_order ){
	order = p_order;
	
	sizeData = correspondTable->size();//size of the table
	PVLOG_INFO("sizeData = %d\n",sizeData);
	sortable.reserve(sizeData);
	
	//prepare data
	for (int idx = 0; idx < sizeData; ++idx) {
		int row = correspondTable->at(idx);
		PVRush::PVNraw::nraw_table_line line = myTable->at(row);//line of log.
		QString *item = new QString(line.at(idColumn));//value at column
		sortable.append(QPair<QString*,int>(item, row));
	}
}
/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringListThread::update
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringListThread::update(){
	PVLOG_INFO("start   update\n");
	//reset the correspond table
	//correspondTable->clear();
	//correspondTable->res
	//update the correspond table
	for (int idx = 0; idx < sizeData; ++idx){
		int key = sortable.at(idx).second;
		correspondTable->at(idx)=key;
		//correspondTable->insert(idx,key);
	}
	PVLOG_INFO("end   update\n");
}
	
	
	
	
	
	
	
