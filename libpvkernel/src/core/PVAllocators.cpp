/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVAllocators.h>

#include <fstream>
#include <iostream> // for basic_istream, ifstream, etc
#include <string>   // for operator>>, string
#include <unistd.h> // for sysconf, _SC_PAGE_SIZE

// Taken from http://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-run-time-in-c
// vm_usage and resident_set are in KB.
void PVCore::PVMemory::get_memory_usage(double& vm_usage, double& resident_set)
{
	vm_usage = 0.0;
	resident_set = 0.0;

	// 'file' stat seems to give the most reliable results
	//
	std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);

	// dummy vars for leading entries in stat that we don't care about
	//
	std::string pid, comm, state, ppid, pgrp, session, tty_nr;
	std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
	std::string utime, stime, cutime, cstime, priority, nice;
	std::string O, itrealvalue, starttime;

	// the two fields we want
	//
	unsigned long vsize;
	long rss;

	stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >>
	    minflt >> cminflt >> majflt >> cmajflt >> utime >> stime >> cutime >> cstime >> priority >>
	    nice >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

	stat_stream.close();

	long page_size_kb =
	    sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
	vm_usage = vsize / 1024.0;
	resident_set = rss * page_size_kb;
}
