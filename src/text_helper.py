from messages import WARNING_DIFFERENT_CHARS

# ====================
def latex_text_display(self,
                       ref: str,
                       hyp: str,
                       features: list,
                       feature_chars: list,
                       chars_per_row: int = 30,
                       num_rows: int = 3):

    strings = {
        'ref': list(self.reference[doc_idx].strip()),
        'hyp': list(self.hypothesis[doc_idx].strip())
    }
    labelled = label_fps_and_fns(
        strings, self.features, self.feature_chars)
    rows = [
        [labelled[i] for i in range(a, b)]
        for (a, b) in zip(
            range(0, chars_per_row*num_rows, chars_per_row),
            range(chars_per_row, chars_per_row*(num_rows+1), chars_per_row)
        )
    ]
    for row in rows:
        row[0] = self.escape_line_end_space(row[0])
        row[-1] = self.escape_line_end_space(row[-1])
    rows = [[self.escape_other_spaces(e) for e in row] for row in rows]
    final_latex = '\n'.join(
        [f"\\texttt{{{''.join(r)}}}\\\\" for r in rows]
    )
    return final_latex


# ====================
def label_fps_and_fns(strings: str, features: list, feature_chars: list):

    output_chars = []
    while strings['ref'] and strings['hyp']:
        features_present = {'ref': [], 'hyp': []}
        next_char = {
            'ref': strings['ref'].pop(0), 'hyp': strings['hyp'].pop(0)}
        try:
            assert next_char['ref'].lower() == next_char['hyp'].lower()
        except AssertionError:
            error_msg = WARNING_DIFFERENT_CHARS.format(
                doc_idx="UNKNOWN",
                ref_str=(next_char['ref'] + ''.join(strings['ref'][:10])),
                hyp_str=(next_char['hyp'] + ''.join(strings['hyp'][:10]))
            )
            print(error_msg)
            return None
        for string in strings.keys():
            if ('CAPITALISATION' in features
               and next_char[string].isupper()):
                features_present[string].append('CAPITALISATION')
            while (len(strings[string]) > 0
                    and strings[string][0] in features):
                features_present[string].append(strings[string].pop(0))
        if 'CAPITALISATION' in features:
            if ('CAPITALISATION' in features_present['ref']
               and 'CAPITALISATION' not in features_present['hyp']):
                output_chars.append(f"\\fn{{{next_char['hyp']}}}")
            elif ('CAPITALISATION' not in features_present['ref']
                    and 'CAPITALISATION' in features_present['hyp']):
                output_chars.append(f"\\fp{{{next_char['hyp']}}}")
            else:
                output_chars.append(next_char['hyp'])
        else:
            output_chars.append(next_char['hyp'])
        for feature in feature_chars:
            if (feature in features_present['ref']
               and feature not in features_present['hyp']):
                output_chars.append(f'\\fn{{\\mbox{{{feature}}}}}')
            elif (feature not in features_present['ref']
                    and feature in features_present['hyp']):
                output_chars.append(f'\\fp{{\\mbox{{{feature}}}}}')
            elif (feature in features_present['ref']
                    and feature in features_present['hyp']):
                output_chars.append(feature)
    return output_chars


# ====================
def escape_line_end_space(entry: str):

    if entry == ' ':
        return r"\Verb+{\ }+"
    else:
        return entry


# ====================
def escape_other_spaces(entry: str):

    if entry == ' ':
        return r"{\ }"
    else:
        return entry
