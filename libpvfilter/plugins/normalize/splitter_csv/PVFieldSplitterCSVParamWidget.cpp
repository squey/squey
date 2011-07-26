#include "PVFieldSplitterCSVParamWidget.h"
#include "PVFieldSplitterCSV.h"
#include <pvfilter/PVFieldsFilter.h>


#include <QLabel>
#include <QVBoxLayout>

#include <QSpacerItem>
#include <QPushButton>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterCSV()))
{
	init();
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::init
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterCSVParamWidget::init()
{
	PVLOG_DEBUG("init PVFieldSplitterCSVParamWidget\n");

	action_menu = new QAction(QString("add CSV Splitter"),NULL);

}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()     start\n");

	//get args
	PVCore::PVArgumentList l =  get_filter()->get_args();


	/*
	 * creating the widget param
	 */
	param_widget = new QWidget();

	//init layout
	QVBoxLayout* layout = new QVBoxLayout(param_widget);
	param_widget->setLayout(layout);
	param_widget->setObjectName("splitter");

	//title
	QLabel* label = new QLabel(tr("CSV"),NULL);
	label->setAlignment(Qt::AlignHCenter);
	layout->addWidget(label);

	//field separator
	QLabel* separator_label = new QLabel(tr("Field separator"),NULL);
	separator_label->setAlignment(Qt::AlignLeft);
	layout->addWidget(separator_label);
	separator_text = new QLineEdit(l["sep"].toString());
	separator_text->setAlignment(Qt::AlignHCenter);
	separator_text->setMaxLength(1);
	layout->addWidget(separator_text);

	//field number of col
	QLabel* col_label = new QLabel(tr("Number of columns"),NULL);
	col_label->setAlignment(Qt::AlignLeft);
	layout->addWidget(col_label);
	child_number_edit = new QLineEdit(QString::number(get_child_count()));
	child_number_org_palette = child_number_edit->palette();
	layout->addWidget(child_number_edit);

	// "set number of childs" button
	QPushButton* set_nchilds_btn = new QPushButton(tr("Set number of columns"));
	layout->addWidget(set_nchilds_btn);

	layout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));

	connect(separator_text, SIGNAL(textChanged(const QString &)), this, SLOT(updateSeparator(const QString &)));
	connect(set_nchilds_btn, SIGNAL(clicked()), this, SLOT(updateNChilds()));

	PVLOG_DEBUG("PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()     end\n");

	return param_widget;
}



/******************************************************************************
 *
 * PVFilter::PVFieldSplitterCSVParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterCSVParamWidget::get_action_menu()
{
	assert(action_menu);
	return action_menu;
}


void PVFilter::PVFieldSplitterCSVParamWidget::updateSeparator(const QString &sep)
{
	QPalette p = separator_text->palette();
	if (sep.size() != 1) {
		p.setColor(QPalette::Window, Qt::red);
		separator_text->setPalette(p);
		return;
	}
	separator_text->setPalette(child_number_org_palette);
	PVCore::PVArgumentList args;
	args["sep"]=QVariant(sep.at(0));
	this->get_filter()->set_args(args);

	emit args_changed_Signal();
}

void PVFilter::PVFieldSplitterCSVParamWidget::updateNChilds()
{
	QString const& txt_n = child_number_edit->text();
	if (txt_n.size() == 0) {
		return;
	}

	set_child_count(txt_n.toLong());
	emit nchilds_changed_Signal();
}

