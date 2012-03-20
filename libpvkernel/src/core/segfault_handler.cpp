#include <pvkernel/core/segfault_handler.h>

#ifdef PICVIZ_DEVELOPER_MODE

#ifndef WIN32
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/sysctl.h>

#include <tbb/atomic.h>

// From http://stackoverflow.com/questions/3596781/detect-if-gdb-is-running
static bool are_we_ptraced()
{
	pid_t pid = fork();
	int status;
	bool res;

	if (pid == -1)
	{
		perror("fork");
		return -1;
	}

	if (pid == 0)
	{
		pid_t ppid = getppid();

		/* Child */
		if (ptrace(PTRACE_ATTACH, ppid, NULL, NULL) == 0)
		{
			/* Wait for the parent to stop and continue it */
			waitpid(ppid, NULL, 0);
			ptrace(PTRACE_CONT, NULL, NULL);

			/* Detach */
			ptrace(PTRACE_DETACH, getppid(), NULL, NULL);

			/* We were the tracers, so gdb is not present */
			res = 0;
		}
		else
		{
			/* Trace failed so gdb is present */
			res = 1;
		}
		exit(res);
	}
	else
	{
		waitpid(pid, &status, 0);
		res = WEXITSTATUS(status);
	}
	return res;
}

// Handle segfault by forking gdb !
// Inspired by http://stackoverflow.com/questions/3151779/how-its-better-to-invoke-gdb-from-program-to-print-its-stacktrace
static void segfault_handler(int sig, siginfo_t* sinfo, void* uctxt)
{
	// We should only accept one call !
	static tbb::atomic<bool> _call;
	if (_call.compare_and_swap(false, true)) {
		return;
	}
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
	// If we are already ptraced (like if gdb is alreadu running on top of us),
	// do nothing here.
	if (are_we_ptraced()) {
		return;
	}

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
