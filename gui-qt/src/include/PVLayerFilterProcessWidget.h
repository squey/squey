#ifndef PVINSPECTOR_PVLAYERFILTERPROCESSWIDGET_H
#define PVINSPECTOR_PVLAYERFILTERPROCESSWIDGET_H

#include <picviz/PVLayerFilter.h>
#include "PVArgumentListWidget.h"
#include "PVTabSplitter.h"

namespace PVInspector {

class PVLayerFilterProcessWidget: public QDialog
{
	Q_OBJECT

public:
	PVLayerFilterProcessWidget(PVTabSplitter* tab, PVCore::PVArgumentList& args, Picviz::PVLayerFilter_p filter_p);
	virtual ~PVLayerFilterProcessWidget();

	void change_args(PVCore::PVArgumentList const& args);

public slots:
	void save_Slot();
	void preview_Slot();
	void cancel_Slot();
	void preset_changed(int currentIndex);

protected:
	void create_btns();
	void set_btns_layout();
	void connect_btns();

	bool process();

private:
	static void process_layer_filter(Picviz::PVLayerFilter* filter, Picviz::PVLayer* layer);

protected:
	PVTabSplitter* _tab;
	Picviz::PVView_p _view;
	Picviz::PVLayerFilter_p _filter_p;
	QLabel* _presets_label;
	QGroupBox* _args_widget_box;
	QHBoxLayout* _presets_layout;
	QComboBox* _presets_combo;
	QPushButton* _cancel_btn;
	QPushButton* _help_btn;
	QPushButton* _preview_btn;
	QPushButton* _apply_btn;
	Picviz::PVLayer _pre_filter_layer_org;
	PVCore::PVArgumentList _args_org;
	PVArgumentListWidget* _args_widget;
	QHBoxLayout* _btn_layout;

private:
	bool _has_apply;
};

}

#endif
