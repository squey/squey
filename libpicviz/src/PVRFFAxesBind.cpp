#include <picviz/PVRFFAxesBind.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVAxisIndexType.h>

Picviz::PVRFFAxesBind::PVRFFAxesBind(PVCore::PVArgumentList const& l)
{
	INIT_FILTER(PVRFFAxesBind, l);
}

DEFAULT_ARGS_FUNC(Picviz::PVRFFAxesBind)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("axis_org", "Axis of original view")].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey("axis_dst", "Axis of final view")].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

void Picviz::PVRFFAxesBind::set_args(PVCore::PVArgumentList const& args)
{
	PVFunctionArgsBase::set_args(args);
	_axis_org = args["axis_org"].value<PVCore::PVAxisIndexType>().get_original_index();
	_axis_dst = args["axis_dst"].value<PVCore::PVAxisIndexType>().get_original_index();
}

void Picviz::PVRFFAxesBind::operator()(PVRow row_org, PVView const& view_org, PVView const& view_dst, PVSelection& sel_dst) const
{
	PVRow nlines_sel = view_dst.get_row_count();

	const PVCore::PVUnicodeString& str = view_org.get_data_unistr_raw(row_org, _axis_org);
	for (PVRow r = 0; r < nlines_sel; r++) {
		if (view_dst.get_data_unistr_raw(r, _axis_dst) == str) {
			sel_dst.set_line(r, true);
		}
	}
}
