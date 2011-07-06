#ifndef PVFILTER_PVFIELDSPLITTERCSV_FILE_H
#define PVFILTER_PVFIELDSPLITTERCSV_FILE_H

#include <pvcore/general.h>
#include <pvcore/PVField.h>
#include <pvfilter/PVFieldsFilter.h>
#include <QByteArray>

extern "C" {
#include "libcsv.h"
}

namespace PVFilter {

class LibExport PVFieldSplitterCSV : public PVFieldsFilter<one_to_many> {
public:
	PVFieldSplitterCSV(PVArgumentList const& args = PVFieldSplitterCSV::default_args());

public:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);
	void set_args(PVArgumentList const& args);

public:
	static void _csv_new_field(void* s, size_t len, void* p);
	static void _csv_new_row(int c, void* p);

protected:
	char _sep;

private:
	struct buf_infos {
		struct csv_parser _p;
		QChar* _f_cur;
		size_t _len_buf;
		PVCore::list_fields *_lf;
		PVCore::list_fields::iterator _it_ins;
		PVCore::PVElement *_parent;
		PVCore::list_fields::size_type _nelts;
		QByteArray _cstr;
	};

	CLASS_FILTER(PVFilter::PVFieldSplitterCSV)
};

}

#endif
