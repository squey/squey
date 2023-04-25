# Copyright (c) 2017 freedesktop-sdk
# Copyright (c) 2018 Codethink Limited
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors:
#        Valentin David <valentin.david@gmail.com>
#        Thomas Coldrick <thomas.coldrick@codethink.co.uk>


"""Flatpak Image Element

A buildstream plugin used to stage its build-dependencies, and metadata
provided by the 'metadata' field in a format useful to generate flatpaks.
"""
import os
from buildstream import utils
from buildstream import Element, ElementError, Scope
import configparser

class FlatpakImageElement(Element):

    BST_STRICT_REBUILD = True
    BST_FORMAT_VERSION = 1

    def configure(self, node):
        self.node_validate(node, [
            'directory', 'include', 'exclude', 'metadata',
            'export'
        ])
        self.directory = self.node_subst_member(node, 'directory')
        export = self.node_subst_member(node, 'export')
        export_abs = os.path.join('/', export)
        self.export = os.path.relpath(export_abs, '/')
        self.include = self.node_get_member(node, list, 'include')
        self.exclude = self.node_get_member(node, list, 'exclude')
        self.metadata = configparser.ConfigParser()
        self.metadata.optionxform = str
        metadata_dict = {}
        for section, pairs in node.get('metadata').items():
            if not section.startswith('__bst'):
                section_dict = {}
                for key in pairs.keys():
                    if not key.startswith('__bst'):
                        section_dict[key] = self.node_subst_member(pairs, key)
                metadata_dict[section] = section_dict

        self.metadata.read_dict(metadata_dict)

    def preflight(self):
        runtime_deps = list(self.dependencies(Scope.RUN, recurse=False))
        if runtime_deps:
            raise ElementError("{}: Only build type dependencies supported by flatpak_image elements"
                               .format(self))

        sources = list(self.sources())
        if sources:
            raise ElementError("{}: flatpak_image elements may not have sources".format(self))

    def get_unique_key(self):
        key = {}
        key['directory'] = self.directory
        key['include'] = sorted(self.include)
        key['exclude'] = sorted(self.exclude)
        key['metadata'] = self.metadata
        key['export'] = self.export
        key['version'] = 11              # Used to force rebuilds after editing the plugin
        return key

    def configure_sandbox(self, sandbox):
        pass

    def stage(self, sandbox):
        pass

    def assemble(self, sandbox):
        self.stage_sources(sandbox, 'input')

        basedir = sandbox.get_directory()
        allfiles = os.path.join(basedir, 'buildstream', 'allfiles')
        reldirectory = os.path.relpath(self.directory, '/')
        subdir = os.path.join(allfiles, reldirectory)
        exportsrcdir = os.path.join(allfiles, self.export)
        etcdir = os.path.join(allfiles, 'etc')
        installdir = os.path.join(basedir, 'buildstream', 'install')
        filesdir = os.path.join(installdir, 'files')
        exportdir = os.path.join(installdir, 'export')
        filesetcdir = os.path.join(filesdir, 'etc')
        stagedir = os.path.join(os.sep, 'buildstream', 'allfiles')

        os.makedirs(allfiles, exist_ok=True)
        os.makedirs(filesdir, exist_ok=True)
        if self.metadata.has_section('Application'):
            os.makedirs(exportdir, exist_ok=True)

        for section in self.metadata.sections():
            if section.startswith('Extension '):
                try:
                    extensiondir = self.metadata.get(section, 'directory')
                    os.makedirs(os.path.join(installdir, 'files', extensiondir), exist_ok=True)
                except PermissionError as e:
                    raise ElementError("Permission denied: Cannot create {}".format(extensiondir))

        with self.timed_activity("Creating flatpak image", silent_nested=True):
            self.stage_dependency_artifacts(sandbox, Scope.BUILD,
                                            path=stagedir,
                                            include=self.include,
                                            exclude=self.exclude)
            utils.link_files(subdir, filesdir)
            if os.path.exists(etcdir):
                utils.link_files(etcdir, filesetcdir)

            if os.path.isdir(exportsrcdir):
                utils.link_files(exportsrcdir, exportdir)
            elif self.metadata.has_section('Application'):
                os.makedirs(exportdir, exist_ok=True)

        metadatafile = os.path.join(installdir, 'metadata')
        with open(metadatafile, "w") as m:
            self.metadata.write(m)
        return os.path.join(os.sep, 'buildstream', 'install')

def setup():
    return FlatpakImageElement
