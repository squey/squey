#include "PVFieldSplitterCSVParamWidget.h"
#include "PVFieldSplitterCSV.h"
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>


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
	_recommands_label = NULL;
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
	QGridLayout* gridLayout = new QGridLayout();
	param_widget->setLayout(layout);
	param_widget->setObjectName("splitter");

	//title
	QLabel* label = new QLabel(tr("CSV"),NULL);
	label->setAlignment(Qt::AlignHCenter);
	layout->addWidget(label);

	//field separator
	QLabel* separator_label = new QLabel(tr("Field separator:"));
	separator_label->setAlignment(Qt::AlignLeft);
	gridLayout->addWidget(separator_label, 0, 0);
	separator_text = new QLineEdit(l["sep"].toString());
	separator_text->setAlignment(Qt::AlignHCenter);
	separator_text->setMaxLength(1);
	gridLayout->addWidget(separator_text, 0, 2);

	//field number of col
	QLabel* col_label = new QLabel(tr("Number of columns:"));
	col_label->setAlignment(Qt::AlignLeft);
	gridLayout->addWidget(col_label, 2, 0);
	child_number_edit = new QLineEdit(QString::number(get_child_count()));
	child_number_org_palette = child_number_edit->palette();
	gridLayout->addWidget(child_number_edit, 2, 2);
	layout->addLayout(gridLayout);
	_recommands_label = new QLabel();
	layout->addWidget(_recommands_label);

	// "set number of children" button
	QPushButton* set_nchilds_btn = new QPushButton(tr("Set number of columns"));
	layout->addWidget(set_nchilds_btn);

	layout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));

	connect(separator_text, SIGNAL(textChanged(const QString &)), this, SLOT(updateSeparator(const QString &)));
	connect(set_nchilds_btn, SIGNAL(clicked()), this, SLOT(updateNChilds()));

	PVLOG_DEBUG("PVFilter::PVFieldSplitterCSVParamWidget::get_param_widget()     end\n");

	update_data_display();

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

	update_recommanded_nfields();

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

// Used below
static bool sort_freq(std::pair<PVCol, PVRow> const& first, std::pair<PVCol, PVRow> const& second)
{
	return first.second > second.second;
}

void PVFilter::PVFieldSplitterCSVParamWidget::update_recommanded_nfields()
{
	PVFilter::PVElementFilterByFields elt_f(_filter->f());
	QStringList const& data = get_data();

	// Compute the frequency of the number of fields with this parameters
	QHash<PVCol, PVRow> freq_fields;
	for (int i = 0; i < data.size(); i++) {
        QString myLine = data[i];
		PVLOG_INFO(qPrintable(myLine+"\n"));
		const QChar* start = myLine.constData();
		QString deep_copy(start, myLine.size());
		PVCore::PVElement elt(NULL, (char*) deep_copy.constData(), (char*) (deep_copy.constData() + myLine.size()));
		// Filter this element
		elt_f(elt);
		if (!elt.valid()) {
			continue;
		}
		PVCol nfields_elt = elt.c_fields().size();
		if (freq_fields.contains(nfields_elt)) {
			freq_fields[nfields_elt]++;
		}
		else {
			freq_fields[nfields_elt] = 1;
		}
	}

	// Sort this
	std::vector<std::pair<PVCol, PVRow> > sorted_freq;
	sorted_freq.reserve(freq_fields.size());
	QHash<PVCol, PVRow>::const_iterator it;
	for (it = freq_fields.begin(); it != freq_fields.end(); it++) {
		sorted_freq.push_back(std::pair<PVCol, PVRow>(it.key(), it.value()));
	}
	std::sort(sorted_freq.begin(), sorted_freq.end(), sort_freq);

	QString txt_info = tr("Recommanded number of fields") + QString(":\n");
	std::vector<std::pair<PVCol, PVRow> >::const_iterator it_fr;
	for (it_fr = sorted_freq.begin(); it_fr != sorted_freq.end(); it_fr++) {
		txt_info += tr("\t%1\t (matches %2% of the elements)").arg(it_fr->first).arg(((double)(it_fr->second)/(double)(data.size()))*100.0);
		txt_info += QString("\n");
	}

	if (_recommands_label) {
		_recommands_label->setText(txt_info);
	}
}

void PVFilter::PVFieldSplitterCSVParamWidget::update_data_display()
{
	update_recommanded_nfields();
}
