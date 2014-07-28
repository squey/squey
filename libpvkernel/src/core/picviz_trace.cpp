
#include <pvkernel/core/picviz_trace.h>

#include <stdio.h>
#include <unistd.h>
#include <execinfo.h>

#define SIZE 200

void PVCore::dump_calltrace()
{
	int s;
	void *buffer[SIZE];

	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
	s = backtrace(buffer, SIZE);
	printf("backtrace() returned %d addresses\n", s);
	backtrace_symbols_fd(buffer, s, STDOUT_FILENO);
}

static void display_size(size_t s)
{
	if (s > (1024 * 1024 * 1024)) {
		printf("%.2f Gio\n", s / (1024. * 1024 * 1024));
	} else if (s > (1024 * 1024)) {
		printf("%.2f Mio\n", s / (1024. * 1024));
	} else if (s > (1024)) {
		printf("%.2f Kio\n", s / 1024.);
	} else {
		printf("%ld o\n", s);
	}
}

void PVCore::dump_meminfo()
{
	printf("##################################################\n");

	FILE* fp = fopen("/proc/self/statm", "r");
	size_t psize = getpagesize();

	size_t m_size, m_resident, m_share, m_text, m_lib;
	fscanf(fp, "%lu%lu%lu%lu%lu", &m_size, &m_resident, &m_share, &m_text, &m_lib);
	fclose(fp);
	(void)m_lib;


	printf("virtual : "); display_size(m_size * psize);
	printf("resident: "); display_size(m_resident * psize);
	printf("share   : "); display_size(m_share * psize);
	printf("text    : "); display_size(m_text * psize);
}
