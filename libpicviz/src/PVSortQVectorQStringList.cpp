//! \file PVSortQVectorQStringList.cpp
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVSortQVectorQStringList.h>


/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringList::PVSortQVectorQStringList
 *
 *****************************************************************************/
Picviz::PVSortQVectorQStringList::PVSortQVectorQStringList(QObject *parent)/*:QDialog()*/{
	/*myThread = new PVSortQVectorQStringListThread(parent);
	dialogBox = new PVInspector::PVProgressBox();
	dialogBox->setMessage(QString(tr("The sort is processing.")));*/
	//setDialog();
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringList::~PVSortQVectorQStringList
 *
 *****************************************************************************/
Picviz::PVSortQVectorQStringList::~PVSortQVectorQStringList(){
	
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringList::setDialog
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringList::setDialog(){
	//set the dialog during the sort
	/*QVBoxLayout *vb = new QVBoxLayout();
	dialogBox.setLayout(vb);
	vb->addWidget(new QLabel("The sort is processing."));
	//set the button to cancel sort.
	QPushButton *btnCancel = new QPushButton(QString("Cancel"));
	vb->addWidget(btnCancel);
	connect(btnCancel,SIGNAL(clicked()),&dialogBox,SLOT(reject()));*/
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringList::setList
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringList::setList(PVRush::PVNraw::nraw_table *dataIn, QVector<int>* correspondingTable){
	//myThread->setList(dataIn,correspondingTable);
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringList::sortByColumn
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringList::sort(int idColumn, Qt::SortOrder p_order ){
	PVLOG_INFO("start sorting\n");
	/*//declare sorting data
	if(this==0){
		PVLOG_WARN("the data can't be sorted. (%s:%d)\n",__FILE__,__LINE__);
		return;
	}
	
	//we use 'QueuedConnection' to keep the dialog visible during the sort.
	connect(myThread,SIGNAL(finished()),this,SLOT(closeDialog()), Qt::QueuedConnection);
	
	myThread->init(idColumn,p_order);
	myThread->start(QThread::LowPriority);

	PVLOG_INFO("waitting : sort processing... \n");
	
	if(dialogBox->exec()){//show dialog and wait for event
		myThread->update();
	}else{//if we cancel during the sort...
		//... no update.
		//... stop the the thread.
		myThread->exit(0);
	}

	

	PVLOG_INFO("end   sorting\n");*/
}

/******************************************************************************
 *
 * Picviz::PVSortQVectorQStringList::closeDialog
 *
 *****************************************************************************/
void Picviz::PVSortQVectorQStringList::closeDialog(){
	//dialogBox->accept();
}

	
	
	
	
	
	
