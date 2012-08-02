/**
 * \file filters.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <apr_general.h>
#include <apr_dso.h>
#include <apr_hash.h>
#include <apr_file_info.h>
#include <apr_strings.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/filters.h>
#include <picviz/utils.h>
#include <picviz/datatreerootitem.h>
#include <picviz/filters.h>
#include <picviz/plugins.h>

/******************************************************************************
 *
 * picviz_filter_plugin_load
 *
 *****************************************************************************/
picviz_filter_t *picviz_filter_plugin_load(apr_pool_t *pool, char *filepath)
{
	apr_dso_handle_t *dso = NULL;
	char errbuf[1024];
	apr_dso_handle_sym_t ressym;
	apr_status_t status;

	picviz_filter_t *filter;

	VM_START  

	filter = (picviz_filter_t *)apr_palloc(pool, sizeof(picviz_filter_t));
	if (!filter) {
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot allocate picviz_filter_t in %s:%s\n", __FILE__, __FUNCTION__);
		return NULL;
	}

	picviz_debug(PICVIZ_DEBUG_NOTICE, "Loading filter plugin '%s'\n", filepath);

	status = apr_dso_load(&filter->dso, filepath, pool);

	status = apr_dso_sym(&ressym, filter->dso, picviz_filtering_function_init_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot get picviz_filtering_function_init symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->init_func = (picviz_filtering_function_init_func)ressym;

	status = apr_dso_sym(&ressym, filter->dso, picviz_filtering_function_get_type_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot get picviz_filtering_function_get_type symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->get_type_func = (picviz_filtering_function_get_type_func)ressym;

	status = apr_dso_sym(&ressym, filter->dso, picviz_filtering_function_get_arguments_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot get picviz_filtering_function_get_arguments symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->get_arguments_func = (picviz_filtering_function_get_arguments_func)ressym;

	status = apr_dso_sym(&ressym, filter->dso, picviz_filtering_function_exec_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot get picviz_filtering_function_exec symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->exec_func = (picviz_filtering_function_exec_func)ressym;

	status = apr_dso_sym(&ressym, filter->dso, picviz_filtering_function_terminate_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot get picviz_filtering_function_terminate symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->terminate_func = (picviz_filtering_function_terminate_func)ressym;	

	VM_END

	return filter;
}



/******************************************************************************
 *
 * picviz_filters_plugin_register_all
 *
 *****************************************************************************/
int picviz_filters_plugin_register_all(apr_pool_t *pool, apr_hash_t *hash)
{
	char *pluginsdir;
	apr_dir_t *dir;
	apr_finfo_t dirent;
	apr_status_t status;
	char *ext;
	char *filepath;
	char *extracted_name;

	picviz_filter_t *filter;

	VM_START


	pluginsdir = picviz_plugins_get_filters_dir();

	/* printf("Functions plugin dir:'%s'\n", pluginsdir); */

	status = apr_dir_open(&dir, pluginsdir, pool);
	if (status != APR_SUCCESS) {
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot open directory '%s'! Unable to load filters plugins.\n", pluginsdir);
		return FALSE;
	}

	while ((apr_dir_read(&dirent, APR_FINFO_NAME, dir)) == APR_SUCCESS) {
		ext = (char *)strrchr(dirent.name, '.');
		if (!ext) continue;
		if (!strcmp(ext, PICVIZ_DLL_EXTENSION)) {
			filepath = apr_pstrcat(pool, pluginsdir, PICVIZ_PATH_SEPARATOR, dirent.name, NULL);
			extracted_name = picviz_utils_plugin_extract_name(filepath);
			filter = picviz_filter_plugin_load(pool, filepath);
			if (!filter) {
				picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot register functions for plugin '%s'\n", filepath);
				continue;
			} else {
				filter->init_func();
				if (hash) {
					extracted_name = strchr(extracted_name, '_');
					extracted_name++;
					apr_hash_set(hash, extracted_name, APR_HASH_KEY_STRING, filter);
				} else {
					picviz_debug(PICVIZ_DEBUG_CRITICAL, "The filters hash is empty!\n");
					return FALSE;
				}
			}
		}
	}

	apr_dir_close(dir);

	VM_END

	return TRUE;
}



/******************************************************************************
 *
 * picviz_filters_foreach_filter
 *
 *****************************************************************************/
void picviz_filters_foreach_filter(picviz_datatreerootitem_t *datatree, picviz_filtering_function_foreach_func foreach_func, void *userdata)
{
	apr_hash_index_t *hi;

	for (hi = apr_hash_first(NULL, datatree->filters_plugin); hi; hi = apr_hash_next(hi)) {
		const char *key;
		char *extracted_name;
		picviz_filter_t *filter;

		apr_hash_this(hi, (const void**)&key, NULL, (void**)&filter);
		foreach_func((char *)key, filter, userdata);
		//printf("ht iteration: key=%s, val=%X\n", key, filter);
	}

}



/******************************************************************************
 *
 * picviz_filters_get_filter_from_name
 *
 *****************************************************************************/
picviz_filter_t *picviz_filters_get_filter_from_name(picviz_datatreerootitem_t *datatree, char *name)
{
	return (picviz_filter_t *)apr_hash_get(datatree->filters_plugin, name, strlen(name));
}
