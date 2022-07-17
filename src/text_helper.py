from misc_helper import check_same_char, display_or_print_html

HTML_STYLE = """<style>
.fp{
    background-color: gray
}
.fn{
    background-color: lightgray
    color: black
}
pre {
  white-space: pre-wrap;
}
</style>"""


# ====================
def latex_text_display(ref: str,
                       hyp: str,
                       features: list,
                       feature_chars: list,
                       start_char: int = 0,
                       chars_per_row: int = 30,
                       num_rows: int = 3):

    chars = {'ref': list(ref), 'hyp': list(hyp)}
    labelled = label_fps_and_fns(chars, features, feature_chars)
    row_char_ranges = zip(
        range(start_char, start_char + chars_per_row*num_rows, chars_per_row),
        range(
            start_char + chars_per_row,
            start_char + (chars_per_row*(num_rows+1)),
            chars_per_row
        )
    )
    rows = [
        [labelled[i] for i in range(a, b)]
        for (a, b) in row_char_ranges
    ]
    rows = [escape_spaces_row(row) for row in rows]
    final_latex = '\n'.join(
        [f"\\texttt{{{''.join(r)}}}\\\\" for r in rows]
    )
    return final_latex


# ====================
def html_text_display(ref: str,
                      hyp: str,
                      features: list,
                      feature_chars: list):

    chars = {'ref': list(ref), 'hyp': list(hyp)}
    labelled = label_fps_and_fns(chars, features,
                                 feature_chars, for_latex=False)
    html = HTML_STYLE + f"<pre>{''.join(labelled)}</pre>"
    display_or_print_html(html)


# ====================
def label_fps_and_fns(chars: str,
                      features: list,
                      feature_chars: list,
                      for_latex: bool = False) -> str:

    output_chars = []
    while chars['ref'] and chars['hyp']:
        next_char = {'ref': chars['ref'].pop(0), 'hyp': chars['hyp'].pop(0)}
        if check_same_char(next_char, chars) is not True:
            return None
        features_present, chars = get_features_present(
            next_char, chars, features)
        output_chars.extend(get_next_entries(
            next_char, features_present, features, feature_chars, for_latex))
    return output_chars


# ====================
def get_features_present(next_char: dict, chars: dict, features: list) -> dict:

    features_present = {'ref': [], 'hyp': []}
    for ref_or_hyp in chars.keys():
        if 'CAPITALISATION' in features and next_char[ref_or_hyp].isupper():
            features_present[ref_or_hyp].append('CAPITALISATION')
        while len(chars[ref_or_hyp]) > 0 and chars[ref_or_hyp][0] in features:
            features_present[ref_or_hyp].append(chars[ref_or_hyp].pop(0))
    return features_present, chars


# ====================
def get_next_entries(next_char: dict,
                     features_present: dict,
                     features: list,
                     feature_chars: list,
                     for_latex: bool = False) -> list:

    class_label = cmd if for_latex else span_class
    char_box = mbox if for_latex else lambda x: x
    next_entries = []
    if 'CAPITALISATION' in features:
        tfpn_ = tfpn('CAPITALISATION', features_present)
        if tfpn_ in ['fn', 'fp']:
            next_entries.append(class_label(tfpn_, next_char['hyp']))
        else:
            next_entries.append(next_char['hyp'])
    else:
        next_entries.append(next_char['hyp'])
    for feature in feature_chars:
        tfpn_ = tfpn(feature, features_present)
        if tfpn_ in ['fn', 'fp']:
            next_entries.append(class_label(tfpn_, char_box(feature)))
        elif tfpn_ == 'tp':
            next_entries.append(feature)
    return next_entries


# ====================
def tfpn(feature: str, features_present: dict) -> str:

    in_ref = feature in features_present['ref']
    in_hyp = feature in features_present['hyp']
    if in_hyp and in_ref:
        return 'tp'
    elif in_ref:
        return 'fn'
    elif in_hyp:
        return 'fp'
    else:
        return 'tn'


# ====================
def span_class(class_: str, char: str) -> str:

    return f'<span class="{class_}">{char}</span>'


# ====================
def cmd(tag: str, char: str) -> str:

    return f"\\{tag}{{{char}}}"


# ====================
def mbox(char: str) -> str:

    return cmd('mbox', char)


# ====================
def fn(char: str) -> str:

    return cmd('fn', char)


# ====================
def fp(char: str) -> str:

    return cmd('fp', char)


# ====================
def escape_spaces_row(row: str) -> str:

    row = escape_row_end_spaces(row)
    row = escape_other_spaces(row)
    return row


# ====================
def escape_row_end_spaces(row: str) -> str:

    row[0] = escape_row_end_space(row[0])
    row[-1] = escape_row_end_space(row[-1])
    return row


# ====================
def escape_row_end_space(entry: str) -> str:

    if entry == ' ':
        return r"\Verb+{\ }+"
    else:
        return entry


# ====================
def escape_other_spaces(row: str) -> str:

    row = [escape_other_space(e) for e in row]
    return row


# ====================
def escape_other_space(entry: str) -> str:

    if entry == ' ':
        return r"{\ }"
    else:
        return entry
