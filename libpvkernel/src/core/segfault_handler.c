#include <pvkernel/core/segfault_handler.h>

#ifdef PICVIZ_DEVELOPER_MODE

#ifndef WIN32
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

// Handle segfault by forking gdb !
// Inspired by http://stackoverflow.com/questions/3151779/how-its-better-to-invoke-gdb-from-program-to-print-its-stacktrace
static void segfault_handler(int sig, siginfo_t* sinfo, void* uctxt)
{
	fprintf(stderr, "/!\\ ----------------------------------------------------------------------------------------------------------- /!\\\n");
	fprintf(stderr, "/!\\ -------- /!\\ Segfault occured at %p, do you want to launch the last-chance gdb ? /!\\ -------- /!\\\n", sinfo->si_addr);
	fprintf(stderr, "/!\\ ----------------------------------------------------------------------------------------------------------- /!\\\n\n");
	fprintf(stderr, "(Y)es/(N)o [Y]: ");
	char line[10];
	line[0] = 0;
	fgets(line, 9, stdin);
	if (line[0] != '\n' && line[0] != 'Y' && line[0] != 'y') {
		abort();
	}

	fprintf(stderr, "Launching gdb...\n");
	// Get the PID as a string
	char pid_buf[11];
	sprintf(pid_buf, "%d", getpid());
	// Get current process path
	// That's linux specific, but, well, we're using gdb :)
	char process_path[1024];
	int ret = readlink("/proc/self/exe", process_path, 1024-1);
	if (ret == -1) {
		perror("Unable to get current process path !");
		// TODO: print backtrace by hand !
		abort();
	}
	process_path[ret] = '\0';
	pid_t child_pid = fork();
	if (!child_pid) {           
		execlp("gdb", "gdb", process_path, pid_buf, NULL);
		// If this execlp fails, it means we're still in the same process. So abort
		fprintf(stderr, "GDB failed to start !\n");
		// TODO: print backtrace by hand !
	} else {
		waitpid(child_pid, NULL, 0);
	}
	abort();
}

void init_segfault_handler()
{
	struct sigaction sa;
	memset(&sa, 0, sizeof(struct sigaction));

	sa.sa_flags = SA_SIGINFO | SA_RESTART;
	sa.sa_sigaction = segfault_handler;

	sigaction(SIGSEGV, &sa, NULL);
}

#else // WIN32
// TODO: provide windows implementation if possible

void init_segfault_handler()
{
}

#endif // WIN32

#else // PICVIZ_DEVELOPER_MODE

// TODO: use google breakpad so that users can send us backtraces ?
void init_segfafult_handler()
{
}

#endif // PICVIZ_DEVELOPER_MODE
