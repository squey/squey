#  Copyright (C) 2018 Abderrahim Kitouni
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this library. If not, see <http://www.gnu.org/licenses/>.
#
#  Authors:
#        Abderrahim Kitouni <akitouni@gnome.org>

"""flatpak repository element

A :mod:`ScriptElement <buildstream.scriptelement>` implementation for exporting
a flatpak repository from a set of `flatpak_image`s.

The flatpak_repo default configuration:
  .. literalinclude:: ../../../bst_external/elements/flatpak_repo.yaml
     :language: yaml
"""

from buildstream import ScriptElement, Scope, ElementError

class FlatpakRepoElement(ScriptElement):
    BST_FORMAT_VERSION = 1
    BST_ARTIFACT_VERSION = 1

    def configure(self, node):
        self.node_validate(node, ['environment', 'copy-refs', 'repo-mode', 'arch', 'branch'])

        self._env = self.node_get_member(node, list, 'environment')

        self._copy_refs = []
        for subnode in self.node_get_member(node, list, 'copy-refs'):
            self.node_validate(subnode, ['src', 'dest'])
            self._copy_refs.append((self.node_subst_member(subnode, 'src'),
                                    self.node_subst_member(subnode, 'dest')))

        self._arch = self.node_subst_member(node, 'arch')
        self._branch = self.node_subst_member(node, 'branch')

        self.set_work_dir()
        self.set_root_read_only(True)

        self._repo_mode = self.node_subst_member(node, 'repo-mode')
        self.set_install_root('/buildstream/repo')
        self.add_commands('init repository',
                          ['ostree init --repo=/buildstream/repo --mode={}'.format(self._repo_mode)])

    def _layout_flatpaks(self, elements):
        def staging_dir(elt):
            return '/buildstream/input/{}'.format(elt.name)

        def export_command(elt):
            return 'flatpak build-export --disable-sandbox --files=files --arch={} /buildstream/repo {} {}'\
                .format(self._arch, staging_dir(elt), self._branch)

        for elt in elements:
            if elt.get_kind() == 'flatpak_image':
                self.layout_add(elt.name, staging_dir(elt))
                self.add_commands('export {}'.format(elt.name), [export_command(elt)])
            elif elt.get_kind() == 'stack':
                self._layout_flatpaks(elt.dependencies(Scope.RUN, recurse=False))
            else:
                raise ElementError('Dependency {} is not of kind flatpak_image'.format(elt.name))

    def stage(self, sandbox):
        env = [self.search(Scope.BUILD, elt) for elt in self._env]
        flatpaks = [elt for elt in self.dependencies(Scope.BUILD, recurse=False) if elt not in env]

        for elt in env:
            self.layout_add(elt.name, '/')

        self._layout_flatpaks(flatpaks)

        for src, dest in self._copy_refs:
            self.add_commands('copy ref {} -> {}'.format(src, dest),
                              ['flatpak build-commit-from --src-ref={} /buildstream/repo {}'.format(src, dest)])

        super(FlatpakRepoElement, self).stage(sandbox)

    def get_unique_key(self):
        return {
            'environment': self._env,
            'copy-refs': self._copy_refs,
            'repo-mode': self._repo_mode,
            'arch': self._arch,
            'branch': self._branch
        }


# Plugin entry point
def setup():
    return FlatpakRepoElement
