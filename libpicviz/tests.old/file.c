#include <stdio.h>
#include <stdlib.h>

#include <picviz/file.h>

void print_file_ending(picviz_file_ending_t ending)
{
	switch (ending) {
		case PICVIZ_FILE_ENDING_UNKNOWN:
			printf("PICVIZ_FILE_ENDING_UNKNOWN\n");
			break;
		case PICVIZ_FILE_ENDING_CR:
			printf("PICVIZ_FILE_ENDING_CR\n");
			break;
		case PICVIZ_FILE_ENDING_LF:
			printf("PICVIZ_FILE_ENDING_LF\n");
			break;
		case PICVIZ_FILE_ENDING_CRLF:
			printf("PICVIZ_FILE_ENDING_CRLF\n");
			break;
		case PICVIZ_FILE_ENDING_LFCR:
			printf("PICVIZ_FILE_ENDING_LFCR\n");
			break;
		default:
			printf("Whaoo, this is not even in unknown!\n");
			exit(1);
	}
}

int test_file(char *filename)
{
	picviz_file_t *file;
	picviz_file_ending_t file_ending;
	
	file = picviz_file_new(filename);
	file_ending = picviz_file_get_ending(file);
	print_file_ending(file_ending);
	picviz_file_destroy(file);

	return file_ending;
}

int file_read_from_end(long offset, char *line, size_t linesize, void *userdata)
{
	printf("line:'%s'\n", line);
	return 0;
}

int main(void)
{
	picviz_file_ending_t retval;
	int exit_code;

	picviz_file_t *file;
	picviz_file_t *copied_file;
	size_t *sizes;
	size_t ref_sizes[3];
	int i;


	exit_code = 0;

	printf("Testing CRLF... ");
	retval = test_file("files/file_ending.crlf");
	if (retval != PICVIZ_FILE_ENDING_CRLF) exit_code = 1;

	printf("Testing LFCR... ");
	retval = test_file("files/file_ending.lfcr");
	if (retval != PICVIZ_FILE_ENDING_LFCR) exit_code = 1;

	printf("Testing LF... ");
	retval = test_file("files/file_ending.lf");
	if (retval != PICVIZ_FILE_ENDING_LF) exit_code = 1;

	printf("Testing CR... ");
	retval = test_file("files/file_ending.cr");
	if (retval != PICVIZ_FILE_ENDING_CR) exit_code = 1;

	printf("Testing UTF16 (ending CRLF)... ");
	retval = test_file("files/file_ending.utf16");
	if (retval != PICVIZ_FILE_ENDING_CRLF) exit_code = 1;

	/* Get line sizes */
	ref_sizes[0] = 17;
	ref_sizes[1] = 45;
	ref_sizes[2] = 5;

	file = picviz_file_new("files/test-sizes");
	sizes = picviz_file_get_lines_size(file, PICVIZ_FILE_READMODE_NORMAL);

	for (i=0; i < file->nblines; i++) {
		printf("Line %d has size %d. Must be %d\n", i+1, sizes[i], ref_sizes[i]);
		if (sizes[i] != ref_sizes[i]) {
			exit_code = 1;
		}
	}
	picviz_file_destroy(file);

	printf("Test the reverse function...\n");
	file = picviz_file_new("files/test.gz");
	file->limit_max = 4;
	printf("This file has %d lines\n", file->nblines);

	picviz_file_line_foreach_reverse(file, file_read_from_end, NULL);
	picviz_file_destroy(file);

	printf("Test the file copy A2B function...\n");
	file = picviz_file_new("files/test.gz");

	copied_file = picviz_file_new_basic();	
	picviz_file_copy_A2B(file, copied_file);
	if (!copied_file) {
	  printf("Unable to copy file!\n");
	  return 1;
	}

	picviz_file_debug(copied_file);

	picviz_file_destroy(file);
	/* picviz_file_destroy(copied_file); */
	
	return exit_code;
}
