from collections import defaultdict
import functools
import os
import random
import sys
import re
from typing import List
import hashlib
import nltk
from fnmatch import fnmatch, fnmatchcase

import rich
# the code below is heavily borrowed from unittest.loader

# what about .pyc (etc)
# we would need to avoid loading the same tests multiple times
# from '.py', *and* '.pyc'
VALID_MODULE_NAME = re.compile(r'[_a-z]\w*\.py$', re.IGNORECASE)


def _jython_aware_splitext(path):
    if path.lower().endswith('$py.class'):
        return path[:-9]
    return os.path.splitext(path)[0]


def three_way_cmp(x, y):
    """Return -1 if x < y, 0 if x == y and 1 if x > y"""
    return (x > y) - (x < y)


class Extraction(object):
    """
    This class is responsible for loading extract methods from a given path.
    """

    def __init__(self, method_name='run_extraction'):
        """Create an instance of the class that will use the named test
           method when executed. Raises a ValueError if the instance does
           not have a method with the specified name.
        """
        self._extract_method_name = method_name

    def id(self):
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}.{self._extract_method_name}"

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented

        return self._extract_method_name == other._extract_method_name

    def __hash__(self):
        return hash((type(self), self._extract_method_name))

    def __str__(self):
        return f"{self._extract_method_name} ({self.__class__.__module__}.{self.__class__.__qualname__})"

    def __repr__(self):
        return f"<{self.__class__.__module__}.{self.__class__.__qualname__} extractMethod={self._extract_method_name}>"

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def run(self, *args, **kwds) -> dict:
        extract_method = getattr(self, self._extract_method_name)
        try:
            args, kwds = self.pre_process(*args, **kwds)
            result = extract_method(*args, **kwds)
            result = self.post_process(result)
        finally:
            pass

        return result

    def pre_process(self, *args, **kwds):
        return args, kwds

    def post_process(self, result: dict) -> dict:
        if not isinstance(result, dict):
            result = {'result': result}
        result['extractor'] = self.id()
        return result


class TreeExtraction(Extraction):

    def pre_process(self, tree: nltk.Tree, tree_id: str = None) -> tuple:
        if isinstance(tree, str):
            try:
                tree = nltk.Tree.fromstring(tree)
            except:
                raise ValueError(f"Cannot convert str {tree} to nltk.Tree")
        elif not isinstance(tree, nltk.Tree):
            raise ValueError(
                f"The input tree must be either a str or nltk.Tree, not {type(tree)}"
            )
        if tree_id is None:
            tree_id = hash(str(tree))
        if tree.label() == '':
            # WSJ trees have an empty label at the root
            tree.set_label('ROOT')
        return [], {'tree_id': tree_id, 'tree': tree}


class QuestionGeneration(Extraction):
    question_builders = None

    def __init__(self, method_name='run_extraction'):
        super().__init__(method_name)
        self.question_templates = {}

    def register_builder(question_type):

        def decorator(func):
            func.__dict__["question_type"] = question_type
            func.__dict__["question_builder"] = True
            return func

        return decorator

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if callable(value) and getattr(value, 'question_builder', False):
            self.load_builders()

    def load_builders(self):
        self.question_builders = defaultdict(list)
        for name in dir(self):
            builder = getattr(self, name)
            if callable(builder) and getattr(builder, 'question_builder',
                                             False):
                self.question_builders[builder.question_type].append(builder)

    @property
    def generate_method_name(self):
        return self._extract_method_name

    def build_options(self, correct_answer, **kwds):
        raise NotImplementedError

    def build_questions(self,
                        ingredient: dict,
                        correct_answer: str,
                        question_tags,
                        question_source,
                        question_infos=None,
                        **kwds):
        if self.question_builders is None:
            self.load_builders()
        instances = []
        for question_type, builders in self.question_builders.items():
            if self.question_templates.get(question_type, None) is None:
                continue
            elif isinstance(self.question_templates[question_type], str):
                question_templates = [self.question_templates[question_type]]
            elif isinstance(self.question_templates[question_type], list):
                question_templates = self.question_templates[question_type]
            else:
                raise ValueError(
                    f"The question template for {question_type} must be either a str or a list of str"
                )
            for question_template in question_templates:
                for builder in builders:
                    # set the random seed the generation of a question is deterministic
                    random.seed(
                        f"{self.generate_method_name} {question_source}")
                    instance = builder(question_template, ingredient,
                                       correct_answer, **kwds)
                    instance['type'] = question_type
                    instance['tags'] = question_tags
                    instance['source'] = question_source
                    instance['question_template'] = question_template
                    instance['md5'] = hashlib.md5(
                        str(instance).encode('utf-8')).hexdigest()
                    if question_infos is not None:
                        instance.update(question_infos)
                    instances.append(instance)
        return instances

    @register_builder('yes_no')
    def build_yes_no_question(self, question_template, ingredient: dict,
                              correct_answer: str, **kwds):
        answer = 'true'
        is_negative = False
        if '<NEG>' in question_template:
            is_negative = True
            question_template = question_template.replace('<NEG>', '')
            question_template = re.sub(r"\s+", ' ', question_template).strip()
        if '{correct_answer}' in question_template:
            question = question_template.format(**ingredient,
                                                correct_answer=correct_answer)
            answer = 'false' if is_negative else 'true'
        elif '{incorrect_answer}' in question_template:
            options = self.build_options(correct_answer, **kwds)
            # remove the correct answer
            options = options[1:]
            question = question_template.format(**ingredient,
                                                incorrect_answer=options[0])
            answer = 'true' if is_negative else 'false'
        else:
            raise ValueError(
                f"The question template [{question_template}] does not contain either {{correct_answer}} or {{incorrect_answer}} markups"
            )

        return {"question": question, "answer": answer}

    @register_builder('multiple_choice')
    def build_multiple_choice_question(self, question_template,
                                       ingredient: dict, correct_answer: str,
                                       **kwds):
        question = question_template.format(**ingredient)
        options = self.build_options(correct_answer, **kwds)
        options, correct_index = self.shuffle_options(options, correct_index=0)

        # add 4% odds of replacing the last option with 'none of the above'
        # the chance of the correct answer is 'none of the above' is 1%
        if random.random() < 0.04:
            # replace 'none of the above' option, it should be the last option
            options[-1] = 'none of the above'
            # the correct answer does not need to change
        return {
            "question": question,
            "options": options,
            "correct_choice": correct_index
        }

    @register_builder('fill_in_the_blank')
    def build_fill_in_the_blank_question(self, question_template,
                                         ingredient: dict, correct_answer: str,
                                         **kwds):
        question = question_template.format(**ingredient)
        return {"question": question, "answer": correct_answer}

    def shuffle_options(self, options: List[str], correct_index: int):
        '''
            Shuffle the options and keep the correct option at the given index
            :param options: a list of options
            :param correct_index: the index of the correct option
            :param seed: the random seed
        '''
        random.seed(
            f"{self.generate_method_name} {options} {self.question_templates}")
        options = options.copy()
        indices = list(range(len(options)))
        random.shuffle(indices)
        shuffled_options = [options[i] for i in indices]
        correct_index = indices.index(correct_index)
        return shuffled_options, correct_index

    def rebuild_text(self, text: str):
        reEMPTY = re.compile(r'<E:[^>:]+(?::(?P<ref_id>\d+))?>')
        # replace the empty nodes
        for match in reEMPTY.finditer(text):
            ref_id = match.group('ref_id')
            if ref_id in self.identity_brackets:
                node_id = self.identity_brackets[ref_id]['id']
                replace_text = self.identity_brackets[ref_id]['plain_text']
                if (node_id in self.json_obj
                        and 'specified_plain_text' in self.json_obj[node_id]):
                    replace_text = self.json_obj[node_id][
                        'specified_plain_text']
                text = text.replace(match.group(0), replace_text)
            else:
                text = text.replace(match.group(0), '')
        text = text.replace(r'\s+', ' ').strip()
        return text

    def pre_process(self, *args, **kwds):
        json_obj = kwds['json_obj']
        self.json_obj = json_obj
        self.tree_index = list(json_obj.keys())[0].split('-')[0]
        self.identity_brackets = json_obj[f"{self.tree_index}-<0>"][
            'identity_brackets']
        self.whole_sentence = json_obj[f"{self.tree_index}-<0>"]['plain_text']
        return args, kwds

    def post_process(self, result: dict) -> dict:
        if not isinstance(result, dict):
            raise ValueError(
                f"The result of {self.generate_method_name} must be a dict")
        if 'instances' not in result:
            raise KeyError(
                f"The result of {self.generate_method_name} must have a key 'instances'"
            )
        for instance in result['instances']:
            instance['sentence'] = self.whole_sentence
            instance['generation_method'] = self.generate_method_name
        # clean up
        del self.json_obj
        del self.tree_index
        del self.identity_brackets
        del self.whole_sentence
        self.question_templates = {}
        return result


class ExtractionSuite(object):
    """
    The suite class that is used to run the extract methods.
    """

    def __init__(self, extractions=()):
        self._extractions = []
        self._removed_extractions = 0
        self.add_extractions(extractions)

    def __repr__(self):
        return f"<{self.__class__.__module__}.{self.__class__.__qualname__} extractions={list(self)}>"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return list(self) == list(other)

    def __iter__(self):
        return iter(self._extractions)

    def add_extraction(self, extraction):
        # sanity checks
        if not callable(extraction):
            raise TypeError("{} is not callable".format(repr(extraction)))
        if isinstance(extraction, type) and issubclass(
                extraction, (Extraction, ExtractionSuite)):
            raise TypeError(
                "Extraction and ExtractionSuite must be instantiated "
                "before passing them to add_extraction()")
        self._extractions.append(extraction)

    def add_extractions(self, extractions):
        if isinstance(extractions, str):
            raise TypeError(
                "extractions must be an iterable of extractions, not a string")
        for extraction in extractions:
            self.add_extraction(extraction)

    def run(self, *args, **kwds):
        results = []
        for index, extraction in enumerate(self):
            result = extraction(*args, **kwds)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)


class ExtractionLoader(object):
    """
    This class is responsible for loading extract methods from a given path.
    This code is inspired by the unittest module.
    """
    extract_method_prefix = 'extract'
    sort_extract_methods_using = staticmethod(three_way_cmp)
    extract_name_patterns = None
    suite_class = ExtractionSuite
    _top_level_dir = None

    def __init__(self):
        super(ExtractionLoader, self).__init__()
        self.errors = []
        # Tracks packages which we have called into via load_tests, to
        # avoid infinite re-entrancy.
        self._loading_packages = set()

    def discover(self, start_dir, pattern='extract*.py', top_level_dir=None):
        """
        Find and return all extract modules from the given start directory.
        """
        set_implicit_top = False
        if top_level_dir is None and self._top_level_dir is not None:
            # make top_level_dir optional if called from load_tests in a package
            top_level_dir = self._top_level_dir
        elif top_level_dir is None:
            set_implicit_top = True
            top_level_dir = start_dir

        top_level_dir = os.path.abspath(top_level_dir)

        if not top_level_dir in sys.path:
            # all test modules must be importable from the top level directory
            # should we *unconditionally* put the start directory in first
            # in sys.path to minimise likelihood of conflicts between installed
            # modules and development versions?
            sys.path.insert(0, top_level_dir)
        self._top_level_dir = top_level_dir

        is_not_importable = False
        is_namespace = False
        extractions = []
        if os.path.isdir(os.path.abspath(start_dir)):
            start_dir = os.path.abspath(start_dir)
            if start_dir != top_level_dir:
                is_not_importable = not os.path.isfile(
                    os.path.join(start_dir, '__init__.py'))
        else:
            # support for discovery from dotted module names
            try:
                __import__(start_dir)
            except ImportError:
                is_not_importable = True
            else:
                the_module = sys.modules[start_dir]
                top_part = start_dir.split('.')[0]
                try:
                    start_dir = os.path.abspath(
                        os.path.dirname((the_module.__file__)))
                except AttributeError:
                    # look for namespace packages
                    try:
                        spec = the_module.__spec__
                    except AttributeError:
                        spec = None

                    if spec and spec.loader is None:
                        if spec.submodule_search_locations is not None:
                            is_namespace = True

                            for path in the_module.__path__:
                                if (not set_implicit_top and
                                        not path.startswith(top_level_dir)):
                                    continue
                                self._top_level_dir = \
                                    (path.split(the_module.__name__
                                         .replace(".", os.path.sep))[0])
                                extractions.extend(
                                    self._find_extractions(path,
                                                           pattern,
                                                           namespace=True))
                    elif the_module.__name__ in sys.builtin_module_names:
                        # builtin module
                        raise TypeError('Can not use builtin modules '
                                        'as dotted module names') from None
                    else:
                        raise TypeError(
                            'don\'t know how to discover from {!r}'.format(
                                the_module)) from None

                if set_implicit_top:
                    if not is_namespace:
                        self._top_level_dir = \
                           self._get_directory_containing_module(top_part)
                        sys.path.remove(top_level_dir)
                    else:
                        sys.path.remove(top_level_dir)

        if is_not_importable:
            raise ImportError('Start directory is not importable: %r' %
                              start_dir)

        if not is_namespace:
            extractions = list(self._find_extractions(start_dir, pattern))
        return self.suite_class(extractions)

    def load_extractions_from_module(self, module, *args, pattern=None, **kws):
        """Return a suite of all test cases contained in the given module"""
        if len(args) > 1:
            # Complain about the number of arguments, but don't forget the
            # required `module` argument.
            complaint = len(args) + 1
            raise TypeError(
                'load_extractions_from_module() takes 1 positional argument but {} were given'
                .format(complaint))
        if len(kws) != 0:
            # Since the keyword arguments are unsorted (see PEP 468), just
            # pick the alphabetically sorted first argument to complain about,
            # if multiple were given.  At least the error message will be
            # predictable.
            complaint = sorted(kws)[0]
            raise TypeError(
                "load_extractions_from_module() got an unexpected keyword argument '{}'"
                .format(complaint))
        extractions = []
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Extraction):
                extractions.append(self.load_extractions_from_extraction(obj))

        load_extractions = getattr(module, 'load_extractions', None)
        extractions = self.suite_class(extractions)
        if load_extractions is not None:
            try:
                return load_extractions(self, extractions, pattern)
            except Exception as e:
                raise ImportError(
                    'Failed to load extraction module: %s' % module.__name__,
                    e)
        return extractions

    def load_extractions_from_extraction(self, extraction_class):
        """Return a suite of all extractions contained in extraction_class"""
        if issubclass(extraction_class, ExtractionSuite):
            raise TypeError("Extractions should not be derived from "
                            "ExtractionSuite. Maybe you meant to derive from "
                            "Extraction?")
        extraction_names = self.get_extraction_names(extraction_class)
        if not extraction_names and hasattr(extraction_class,
                                            'run_extraction'):
            extraction_names = ['run_extraction']
        loaded_suite = self.suite_class(map(extraction_class,
                                            extraction_names))
        return loaded_suite

    def get_extraction_names(self, extraction_class):
        """Return a sorted sequence of method names found within testCaseClass
        """

        def should_include_method(attrname):
            if not attrname.startswith(self.extract_method_prefix):
                return False
            extract_func = getattr(extraction_class, attrname)
            if not callable(extract_func):
                return False
            full_name = f'%s.%s.%s' % (extraction_class.__module__,
                                       extraction_class.__qualname__, attrname)
            return self.extract_name_patterns is None or \
                any(fnmatchcase(full_name, pattern) for pattern in self.extract_name_patterns)

        extract_fn_names = list(
            filter(should_include_method, dir(extraction_class)))
        if self.sort_extract_methods_using:
            extract_fn_names.sort(
                key=functools.cmp_to_key(self.sort_extract_methods_using))
        return extract_fn_names

    def _get_module_from_name(self, name):
        __import__(name)
        return sys.modules[name]

    def _get_name_from_path(self, path):
        if path == self._top_level_dir:
            return '.'
        path = _jython_aware_splitext(os.path.normpath(path))

        _relpath = os.path.relpath(path, self._top_level_dir)
        assert not os.path.isabs(_relpath), "Path must be within the project"
        assert not _relpath.startswith('..'), "Path must be within the project"

        name = _relpath.replace(os.path.sep, '.')
        return name

    def _match_path(self, path, full_path, pattern):
        # override this method to use alternative matching strategy
        return fnmatch(path, pattern)

    def _find_extractions(self, start_dir, pattern, namespace=False):
        """Used by discovery. Yields test suites it loads."""
        # Handle the __init__ in this package
        name = self._get_name_from_path(start_dir)
        # name is '.' when start_dir == top_level_dir (and top_level_dir is by
        # definition not a package).
        if name != '.' and name not in self._loading_packages:
            # name is in self._loading_packages while we have called into
            # loadTestsFromModule with name.
            extractions, should_recurse = self._find_extraction_path(
                start_dir, pattern, namespace)
            if extractions is not None:
                yield extractions
            if not should_recurse:
                # Either an error occurred, or load_tests was used by the
                # package.
                return
        # Handle the contents.
        paths = sorted(os.listdir(start_dir))
        for path in paths:
            full_path = os.path.join(start_dir, path)
            extractions, should_recurse = self._find_extraction_path(
                full_path, pattern, namespace)
            if extractions is not None:
                yield extractions
            if should_recurse:
                # we found a package that didn't use load_tests.
                name = self._get_name_from_path(full_path)
                self._loading_packages.add(name)
                try:
                    yield from self._find_extractions(full_path, pattern,
                                                      namespace)
                finally:
                    self._loading_packages.discard(name)

    def _find_extraction_path(self, full_path, pattern, namespace=False):
        """Used by discovery.

        Loads tests from a single file, or a directories' __init__.py when
        passed the directory.

        Returns a tuple (None_or_tests_from_file, should_recurse).
        """
        basename = os.path.basename(full_path)
        if os.path.isfile(full_path):
            if not VALID_MODULE_NAME.match(basename):
                # valid Python identifiers only
                return None, False
            if not self._match_path(basename, full_path, pattern):
                return None, False
            # if the test file matches, load it
            name = self._get_name_from_path(full_path)
            try:
                module = self._get_module_from_name(name)
            except:
                raise ImportError('Failed to import extraction module: %s' %
                                  name)
            else:
                mod_file = os.path.abspath(
                    getattr(module, '__file__', full_path))
                realpath = _jython_aware_splitext(os.path.realpath(mod_file))
                fullpath_noext = _jython_aware_splitext(
                    os.path.realpath(full_path))
                if realpath.lower() != fullpath_noext.lower():
                    module_dir = os.path.dirname(realpath)
                    mod_name = _jython_aware_splitext(
                        os.path.basename(full_path))
                    expected_dir = os.path.dirname(full_path)
                    msg = ("%r module incorrectly imported from %r. Expected "
                           "%r. Is this module globally installed?")
                    raise ImportError(msg %
                                      (mod_name, module_dir, expected_dir))
                return self.load_extractions_from_module(
                    module, pattern=pattern), False
        elif os.path.isdir(full_path):
            if (not namespace and not os.path.isfile(
                    os.path.join(full_path, '__init__.py'))):
                return None, False

            load_extractions = None
            extractions = None
            name = self._get_name_from_path(full_path)
            try:
                package = self._get_module_from_name(name)
            except:
                raise ImportError('Failed to import extraction module: %s' %
                                  name)
            else:
                load_extractions = getattr(package, 'load_extractions', None)
                # Mark this package as being in load_extractions (possibly ;))
                self._loading_packages.add(name)
                try:
                    extractions = self.load_extractions_from_module(
                        package, pattern=pattern)
                    if load_extractions is not None:
                        # loadTestsFromModule(package) has loaded tests for us.
                        return extractions, False
                    return extractions, True
                finally:
                    self._loading_packages.discard(name)
        else:
            return None, False
