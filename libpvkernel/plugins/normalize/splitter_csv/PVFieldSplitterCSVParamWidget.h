#ifndef PVFIELDSPLITTERCSVPARAMWIDGET_H
#define PVFIELDSPLITTERCSVPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

#include <QBoxLayout>
#include <QLabel>
#include <QObject>
#include <QAction>
#include <QLineEdit>

#include "qkeysequencewidget/qkeysequencewidget.h"

namespace PVFilter {

class PVFieldSplitterCSVParamWidget: public PVFieldsSplitterParamWidget {
	Q_OBJECT

public:

	PVFieldSplitterCSVParamWidget();
	//PVFieldSplitterCSVParamWidget(const PVFieldSplitterCSVParamWidget& src);

private:
	QAction* action_menu;
	QWidget* param_widget;
	QLineEdit *child_number_edit;
	QPalette child_number_org_palette;
	QKeySequenceWidget* separator_text;
	//QLineEdit* separator_text;
	QLabel* _recommands_label;
	int id;

public:
	QWidget* get_param_widget();
	QAction* get_action_menu();

	void set_id(int id_param) {
		id = id_param;
	}
	void update_data_display();

private:
	void init();
	void update_recommanded_nfields();


	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterCSVParamWidget)

public slots:
	//void updateSeparator(const QString &sep);
	void updateSeparator(QKeySequence key);
	void updateNChilds();
	

signals:
	void signalRefreshView();
};

}

#endif
