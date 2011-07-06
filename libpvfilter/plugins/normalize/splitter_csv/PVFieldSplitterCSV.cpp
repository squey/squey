#include "PVFieldSplitterCSV.h"


PVFilter::PVFieldSplitterCSV::PVFieldSplitterCSV(PVArgumentList const& args)
{
	INIT_FILTER(PVFilter::PVFieldSplitterCSV, args);

}

void PVFilter::PVFieldSplitterCSV::set_args(PVArgumentList const& args)
{
	FilterT::set_args(args);
	_sep = args["sep"].toChar().toAscii();
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterCSV)
{
	PVArgumentList args;
	args["sep"] = QVariant(',');
	return args;
}

void PVFilter::PVFieldSplitterCSV::_csv_new_field(void* s, size_t len, void* p)
{
	// WARNING: this function will be called from different thread, so we can't use member
	// variables to store informations between different calls...
		
	// TOFIX: All of this is clearly suboptimal, too many memory copy occurs...
	// The ideal fix would be to have just an index and len to the original pointer !
	// (but this works)
	
	buf_infos *sp = (buf_infos*) p;
	QByteArray ba((const char*)s, len);
	QString str_field(ba); // convert to unicode

	size_t len_qs = str_field.size() * sizeof(QChar);
	size_t nchars = str_field.size(); 
	if (len_qs > sp->_len_buf) {
		PVLOG_WARN("(PVFieldSplitterCSV) chunk is not large enough to hold CSV datas !\n");
		return;
	}

	memcpy(sp->_f_cur, str_field.constData(), len_qs);
	PVCore::PVField f(*sp->_parent, (char*) (sp->_f_cur), (char*) (sp->_f_cur+nchars));

	sp->_f_cur += nchars;
	sp->_len_buf -= len_qs;

	PVLOG_HEAVYDEBUG("(PVFieldSplitterCSV): new CSV field: %s\n", qPrintable(f.qstr()));

	sp->_lf->insert(sp->_it_ins, f);
	sp->_nelts++;
}

void PVFilter::PVFieldSplitterCSV::_csv_new_row(int /*c*/, void* /*p*/)
{
	// Should only happen once at most !		
	PVLOG_DEBUG("(PVCore::PVFieldSplitterCSV) in csv_new_row\n");
}

PVCore::list_fields::size_type PVFilter::PVFieldSplitterCSV::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	PVLOG_HEAVYDEBUG("(PVFieldSplitterCSV): in one_to_many\n");

	field.init_qstr();

	buf_infos inf;
	// Init csv parser
	if (csv_init(&inf._p, 0) != 0)
		PVLOG_ERROR("Unable to initialize libcsv !\n");
	csv_set_delim(&inf._p, _sep);

	// Convert to "C strings"
	inf._parent = field.elt_parent();
	inf._it_ins = it_ins;
	inf._lf = &l;
	inf._f_cur = (QChar*) field.begin();
	inf._len_buf = field.size();
	inf._cstr = field.qstr().toAscii();
	inf._nelts = 0;

	// And use libcsv
	csv_parse(&inf._p, inf._cstr.data(), inf._cstr.size(), &PVFilter::PVFieldSplitterCSV::_csv_new_field, &PVFilter::PVFieldSplitterCSV::_csv_new_row, &inf);
	csv_fini(&inf._p, &PVFilter::PVFieldSplitterCSV::_csv_new_field, &PVFilter::PVFieldSplitterCSV::_csv_new_row, &inf);
	csv_free(&inf._p);

	return inf._nelts;
}

IMPL_FILTER(PVFilter::PVFieldSplitterCSV)
