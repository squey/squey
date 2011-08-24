#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LINE 1024*1024

void usage(char* path)
{
	std::cerr << "Usage: " << path << " host port task_nb final file" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 5) {
		usage(argv[0]);
		return 1;
	}

	char* host = argv[1];
	uint16_t port = atoi(argv[2]);
	char* task_id = argv[3];
	bool final = (argv[4][0] == '1');
	if (!final && argc < 6) {
		usage(argv[0]);
		return 1;
	}
	char* file;
	if (!final) {
		file = argv[5];
	}

	FILE* f;
	if (!final) {
		f = fopen(file, "r");
		if (f == NULL) {
			std::cerr << "Unable to open file " << file << std::endl;
			return 1;
		}
	}

	struct hostent* hp;
	hp = gethostbyname(host);
	if (hp == NULL) {
		std::cerr << "Unable to resolve " << host << std::endl;
		return 1;
	}

	struct sockaddr_in sin;
	memset(&sin, 0, sizeof(sin));
	memcpy(&sin.sin_addr, hp->h_addr, hp->h_length);
	sin.sin_family = hp->h_addrtype;
	sin.sin_port = htons(port);

	int s = socket(AF_INET, SOCK_STREAM, 0);
	if (s <= 0) {
		std::cerr << "Unable to create a TCP socket !" << std::endl;
		return 1;
	}

	int r = connect(s, (struct sockaddr*) &sin, sizeof(struct sockaddr_in));
	if (r < 0) {
		std::cerr << "Unable to connect to " << host << ":" << port << std::endl;
		return 1;
	}

	// Write the task number
	write(s, task_id, strlen(task_id));
	write(s, "\n", 1);

	// Is that the final task ?
	write(s, final ? "1":"0", 1);
	write(s, "\n", 1);

	if (!final) {
		// Write down the data, one element per line, and one field per element
		char buf[MAX_LINE+1];
		uint64_t off = 0;
		while ((fgets(buf, MAX_LINE, f)) != NULL) {
			uint32_t sline = strlen(buf);
			uint32_t selt = sline+4;
			write(s, &off, sizeof(uint64_t));
			write(s, &selt, sizeof(uint32_t));
			write(s, &sline, sizeof(uint32_t));
			write(s, buf, sline);
			off = ftell(f);
		}
		fclose(f);
	}

	close(s);

	return 0;
}
