from ruamel import yaml
import os
from buildstream import Element, ElementError, Scope
import collections

class SnapImageElement(Element):

    def configure(self, node):
        self.node_validate(node, [
            'directory', 'include', 'exclude', 'metadata'
        ])
        self.directory = self.node_subst_member(node, 'directory')
        self.include = self.node_get_member(node, list, 'include')
        self.exclude = self.node_get_member(node, list, 'exclude')
        self.metadata = self._clean_meta_data(node.get('metadata'))

    def _clean_meta_data(self, node):
        ret = {}
        for k, v in node.items():
            if not k.startswith('__bst'):
                if isinstance(v, collections.Mapping):
                    ret[k] = self._clean_meta_data(v)
                elif isinstance(v, list):
                    ret[k] = self.node_subst_list(node, k)
                else:
                    ret[k] = self.node_subst_member(node, k)
        return ret

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
        key['version'] = 7
        return key

    def configure_sandbox(self, sandbox):
        pass

    def stage(self, sandbox):
        pass

    def assemble(self, sandbox):
        basedir = sandbox.get_directory()

        reldirectory = os.path.relpath(self.directory, os.sep)
        rootdir = os.path.join(basedir, reldirectory)

        metadir = os.path.join(rootdir, 'meta')
        metadata = os.path.join(metadir, 'snap.yaml')

        with self.timed_activity("Creating snap image", silent_nested=True):
            self.stage_dependency_artifacts(sandbox,
                                            Scope.BUILD,
                                            include=self.include,
                                            exclude=self.exclude)

            os.makedirs(metadir, exist_ok=True)

            with open(metadata, 'w') as f:
                yaml.dump(self.metadata, f)


        return os.path.join(os.sep, reldirectory)

def setup():
    return SnapImageElement
