"""Writing of config files that keeps the help texts they carry.

:class:`configparser.ConfigParser` reads the comments of a file but does not keep them, so
writing a loaded parser back over its own file strips every help text and every blank line the
config had. LEDSA saves the config whenever it fills in a value it asked for or read from an
image, which left the user with a config that no longer explains any of its options.

:func:`save_config` writes the file in place instead: the lines of the old file are kept and
only the values of the options are replaced with the ones the parser holds now. Options and
sections the parser has gained in the meantime are appended, options it no longer holds are
dropped.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

COMMENT_PREFIXES = ('#', ';')


def save_config(parser, file_path: str) -> None:
    """Write *parser* to *file_path*, keeping the comments and the layout of an existing file.

    A file that does not exist yet is written by :meth:`configparser.RawConfigParser.write`,
    which gives it the help texts the config template stores as options without a value.

    :param parser: The configuration to write.
    :type parser: configparser.RawConfigParser
    :param file_path: Path of the config file.
    :type file_path: str
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as config_file:
            parser.write(config_file)
        return
    with open(file_path, encoding='utf-8-sig') as config_file:
        old_lines = config_file.read().splitlines()
    new_lines = _merge_into_lines(parser, old_lines)
    with open(file_path, 'w', encoding='utf-8') as config_file:
        config_file.writelines(line + '\n' for line in new_lines)


def _merge_into_lines(parser, old_lines: Sequence[str]) -> List[str]:
    """Return the lines of the config file with the values *parser* holds filled in."""
    lines = []
    written_sections = []
    for section, section_lines in _split_into_sections(old_lines):
        if section is None:  # the comments above the first section header
            lines.extend(section_lines)
        elif section == parser.default_section or parser.has_section(section):
            lines.extend(_merge_section(parser, section, section_lines))
            written_sections.append(section)
    for section in _sections(parser):
        if section not in written_sections:
            if lines and lines[-1].strip():
                lines.append('')  # a blank line separates it from the section above
            lines.extend(_new_section_lines(parser, section))
    return lines


def _split_into_sections(old_lines: Sequence[str]) -> List[Tuple[Optional[str], List[str]]]:
    """Split the file into the lines above the first section header and one block per section.

    :return: ``(section name, lines)`` pairs, the name of the first block being None. Each
        block starts with its section header.
    """
    blocks: List[Tuple[Optional[str], List[str]]] = [(None, [])]
    for line in old_lines:
        text = line.strip()
        if text.startswith('[') and text.endswith(']'):
            blocks.append((text[1:-1].strip(), [line]))
        else:
            blocks[-1][1].append(line)
    return blocks


def _merge_section(parser, section: str, section_lines: Sequence[str]) -> List[str]:
    """Return the lines of one section block with the values *parser* holds filled in.

    The comments, the blank lines and the spelling of the option names are taken from the file,
    the values from the parser. Options the parser holds that the block does not name are
    appended at its end, in front of the blank lines separating it from the next section.
    """
    options = _options_of_section(parser, section)
    lines = [section_lines[0]]
    written = set()  # the keys of the options the block names
    indent_level = 0
    in_value = False
    for line in section_lines[1:]:
        if not line.strip() or _is_comment(line):
            lines.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if in_value and indent > indent_level:
            continue  # a line of a value that has just been written as a whole
        indent_level = indent
        in_value = False
        name = _option_name(parser, line)
        key = _key(parser, name)
        if key not in options:
            continue  # the option is not part of the configuration any more
        written.add(key)
        in_value = True
        lines.append(_option_line(parser, line[:indent] + name, options[key][1]))
    end = len(lines)
    while end > 1 and not lines[end - 1].strip():
        end -= 1
    lines[end:end] = [_option_line(parser, name, value)
                      for key, (name, value) in options.items() if key not in written]
    return lines


def _new_section_lines(parser, section: str) -> List[str]:
    """Render a section the file does not hold yet, with the comments the parser carries."""
    lines = ['[{}]'.format(section)]
    lines.extend(_option_line(parser, name, value) for name, value in _items(parser, section))
    lines.append('')
    return lines


def _options_of_section(parser, section: str) -> Dict[str, Tuple[str, Optional[str]]]:
    """Return the options written in *section* itself as ``{key: (name, value)}``.

    The keys are stripped of the indentation the config template gives the option names, so
    that an option line of the file can be looked up by the name it gives. The comments the
    template stores as options without a value are left out, because a file that was read
    carries them as comment lines, which are kept as they are.
    """
    options = {}
    for name, value in _items(parser, section):
        if not name.strip() or _is_comment(name):
            continue
        options[_key(parser, name)] = (name, value)
    return options


def _items(parser, section: str) -> List[Tuple[str, Optional[str]]]:
    """Return the ``(name, value)`` pairs of *section*, without the ones inherited from DEFAULT.

    An option a section sets to another value than the DEFAULT section is one of its own, so
    the mapping the parser keeps per section is read directly.
    """
    if section == parser.default_section:
        return list(parser.defaults().items())
    try:
        return list(parser._sections[section].items())
    except (AttributeError, KeyError):
        # Without that mapping, an option overriding a DEFAULT value cannot be told from an
        # inherited one, so only the options no DEFAULT value is known for are left.
        defaults = parser.defaults()
        return [(name, parser.get(section, name, raw=True))
                for name in parser[section] if name not in defaults]


def _sections(parser) -> List[str]:
    """Return the names of every section of *parser*, the DEFAULT section first."""
    sections = [parser.default_section] if parser.defaults() else []
    sections.extend(parser.sections())
    return sections


def _option_name(parser, line: str) -> str:
    """Return the name an option line gives, i.e. the text in front of its first delimiter."""
    text = line.strip()
    positions = [text.find(delimiter) for delimiter in _delimiters(parser)]
    positions = [position for position in positions if position > 0]
    return text[:min(positions)].rstrip() if positions else text


def _option_line(parser, name: str, value: Optional[str]) -> str:
    """Render one option the way the config file holds it.

    The lines of a value spanning multiple lines are indented deeper than its option name, so
    that reading the file gives the value back instead of taking a line of it for an option.
    """
    if value is None:
        return name
    indent = len(name) - len(name.lstrip())
    return '{}{}{}'.format(name, _delimiter(parser),
                           str(value).replace('\n', '\n\t' + ' ' * indent))


def _key(parser, name: str) -> str:
    """Return the name of an option in the form the parser and the file can be matched by."""
    return parser.optionxform(name.strip())


def _delimiters(parser) -> Sequence[str]:
    """Return the characters separating an option name from its value."""
    return getattr(parser, '_delimiters', ('=', ':'))


def _delimiter(parser) -> str:
    """Return the delimiter written between an option name and its value."""
    return ' {} '.format(_delimiters(parser)[0])


def _is_comment(text: str) -> bool:
    """Tell whether a line of the file or an option name is a comment."""
    return text.strip().startswith(COMMENT_PREFIXES)
