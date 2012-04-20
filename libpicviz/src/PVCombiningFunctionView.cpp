#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVTFViewRowFiltering.h>

Picviz::PVCombiningFunctionView::PVCombiningFunctionView()
{
	// AG: for now, the only TF in here is a row filtering one
	PVTransformationFunctionView_p tf_p(new PVTFViewRowFiltering());
	_tfs.push_back(tf_p);
}

void Picviz::PVCombiningFunctionView::pre_process(PVView const& view_src, PVView const& view_dst)
{
	foreach(PVTransformationFunctionView_p const& tf, _tfs) {
		tf->pre_process(view_src, view_dst);
	}
}

Picviz::PVSelection Picviz::PVCombiningFunctionView::operator()(PVView const& view_src, PVView const& view_dst) const
{
	// AG: this is now hard-coded in here, the idea is to have the user being able to modify this in a close future...
	Picviz::PVSelection const& sel_src = view_src.get_real_output_selection();
	if (_tfs.size() == 0) {
		return sel_src;
	}

	std::vector<PVSelection> out_sel;
	out_sel.reserve(_tfs.size());
	foreach(PVTransformationFunctionView_p const& tf, _tfs) {
		out_sel.push_back((*tf)(view_src, view_dst, sel_src));
	}

	// Merge with an OR operation
	// For instance, the user could choose the operation he wants to do here !
	Picviz::PVSelection& ret(out_sel.front()); 
	std::vector<PVSelection>::const_iterator it_sel = out_sel.begin();
	it_sel++;
	for (; it_sel != out_sel.end(); it_sel++) {
		ret |= *it_sel;
	}

	return ret;
}

Picviz::PVTFViewRowFiltering* Picviz::PVCombiningFunctionView::get_first_tf()
{
	return dynamic_cast<PVTFViewRowFiltering*>(_tfs[0].get());
}
