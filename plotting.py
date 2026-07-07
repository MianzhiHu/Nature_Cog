import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape as xml_escape

# ======================================================================================================================
# Load the data
# ======================================================================================================================
E1_dm_switch = pd.read_csv('./data/dm_switch.csv')
E1_model_params = pd.read_csv('./data/dm_summary_modeled.csv')
EV_history_emmeans = pd.read_csv('./data/EV_history_emmeans.csv')
rank_2_emmeans = pd.read_csv('./data/rank_2_emmeans.csv')
E2_dm_switch = pd.read_csv('./data/E2_dm_switch.csv')
E2_model_params = pd.read_csv('./data/E2_dm_summary_modeled.csv')
E2_EV_history_emmeans = pd.read_csv('./data/E2_EV_history_emmeans.csv')
E2_rank_2_emmeans = pd.read_csv('./data/E2_rank_2_emmeans.csv')

font_path = 'utils/AbhayaLibre-ExtraBold.ttf'
prop = fm.FontProperties(fname=font_path)
palette = sns.color_palette('deep')
nature_color = palette[2]
urban_color = palette[3]
control_color = palette[7]
condition_order = ['Nature', 'Urban', 'Control']
task_order = ['First', 'Second']
palette_custom = [nature_color, urban_color, control_color]
condition_palette = {'Nature': nature_color, 'Urban': urban_color, 'Control': control_color}
task_palette = {'First': urban_color, 'Second': nature_color}
strategy_order = ['Exploitation', 'Exploration']
strategy_palette = {'Exploitation': control_color, 'Exploration': palette[0]}

model_parameter_cols = ['t', 'dis_sd', 'noise_sd', 'decay', 'decay_center']
model_parameter_labels = {
    't': 'Inverse Temperature',
    'dis_sd': 'Reward Variance',
    'noise_sd': 'Noise Variance',
    'decay': 'Decay Rate',
    'decay_center': 'Decay Center',
}
model_parameter_docx_path = './data/E1_best_fitting_model_parameter_table.docx'


def _word_escape(value):
    if pd.isna(value):
        return ''
    return xml_escape(str(value))


def _word_run(text, size=22, bold=False, color='000000'):
    bold_tag = '<w:b/>' if bold else ''
    return (
        '<w:r>'
        '<w:rPr>'
        '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>'
        f'<w:color w:val="{color}"/>'
        f'<w:sz w:val="{size}"/>'
        f'<w:szCs w:val="{size}"/>'
        f'{bold_tag}'
        '</w:rPr>'
        f'<w:t>{_word_escape(text)}</w:t>'
        '</w:r>'
    )


def _word_paragraph(text, style=None, size=22, bold=False, color='000000',
                    before=0, after=120, line=300, align=None):
    style_tag = f'<w:pStyle w:val="{style}"/>' if style else ''
    align_tag = f'<w:jc w:val="{align}"/>' if align else ''
    return (
        '<w:p>'
        '<w:pPr>'
        f'{style_tag}'
        f'{align_tag}'
        f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'
        '</w:pPr>'
        f'{_word_run(text, size=size, bold=bold, color=color)}'
        '</w:p>'
    )


def _word_table_cell(text, width, header=False, align='center'):
    text_size = 17 if header else 18
    return (
        '<w:tc>'
        '<w:tcPr>'
        f'<w:tcW w:w="{width}" w:type="dxa"/>'
        '<w:vAlign w:val="center"/>'
        '</w:tcPr>'
        f'{_word_paragraph(text, size=text_size, bold=header, before=0, after=0, line=240, align=align)}'
        '</w:tc>'
    )


def _word_table(rows, widths):
    table_width = sum(widths)
    border = '<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    border += '<w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    border += '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    border += '<w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    border += '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    border += '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'

    table = [
        '<w:tbl>',
        '<w:tblPr>',
        f'<w:tblW w:w="{table_width}" w:type="dxa"/>',
        '<w:tblInd w:w="120" w:type="dxa"/>',
        '<w:tblLayout w:type="fixed"/>',
        f'<w:tblBorders>{border}</w:tblBorders>',
        '<w:tblCellMar>'
        '<w:top w:w="80" w:type="dxa"/>'
        '<w:start w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/>'
        '<w:end w:w="120" w:type="dxa"/>'
        '</w:tblCellMar>',
        '</w:tblPr>',
        '<w:tblGrid>',
        ''.join(f'<w:gridCol w:w="{width}"/>' for width in widths),
        '</w:tblGrid>',
    ]

    for row_idx, row in enumerate(rows):
        header = row_idx == 0
        cells = []
        for col_idx, cell_value in enumerate(row):
            align = 'left' if col_idx == 0 and not header else 'center'
            cells.append(_word_table_cell(cell_value, widths[col_idx], header=header, align=align))
        table.append(f'<w:tr>{"".join(cells)}</w:tr>')

    table.append('</w:tbl>')
    return ''.join(table)


def _word_styles_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="120" w:line="300" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="360" w:after="200" w:line="300" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:color w:val="000000"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="280" w:after="140" w:line="300" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:color w:val="000000"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="200" w:after="100" w:line="300" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:color w:val="000000"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
</w:styles>'''


def _write_model_parameter_docx(model_param_table, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table_widths = [1200, 1500, 600, 1932, 1932, 1932, 1932, 1932]
    table_header = ['Condition', 'Model', 'N'] + [model_parameter_labels[param] for param in model_parameter_cols]
    body = [
        _word_paragraph(
            'E1 Best-Fitting Model Parameters',
            style='Heading1',
            size=28,
            bold=True,
            color='000000',
            before=360,
            after=160,
        ),
        _word_paragraph(
            'Participant-level best-fitting model parameters summarized as mean +/- 1 SD.',
            size=22,
            before=0,
            after=120,
            line=300,
        ),
    ]

    for task in task_order:
        task_data = model_param_table[model_param_table['Task'] == task].copy()
        if task_data.empty:
            continue
        task_data = task_data.sort_values('Condition')
        body.append(
            _word_paragraph(
                f'{task} Task',
                style='Heading2',
                size=24,
                bold=True,
                color='000000',
                before=240,
                after=80,
            )
        )
        rows = [table_header]
        for _, row in task_data.iterrows():
            rows.append(
                [row['Condition'], row['Model'], int(row['N'])]
                + [row[model_parameter_labels[param]] for param in model_parameter_cols]
            )
        body.append(_word_table(rows, table_widths))

    section_props = (
        '<w:sectPr>'
        '<w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:space="720"/>'
        '<w:docGrid w:linePitch="360"/>'
        '</w:sectPr>'
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        f'<w:body>{"".join(body)}{section_props}</w:body>'
        '</w:document>'
    )

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    package_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    core_props = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>E1 Best-Fitting Model Parameters</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''
    app_props = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>'''

    with ZipFile(output_path, 'w', ZIP_DEFLATED) as docx:
        docx.writestr('[Content_Types].xml', content_types)
        docx.writestr('_rels/.rels', package_rels)
        docx.writestr('word/document.xml', document_xml)
        docx.writestr('word/styles.xml', _word_styles_xml())
        docx.writestr('docProps/core.xml', core_props)
        docx.writestr('docProps/app.xml', app_props)


def apply_e1_plot_style(ax, ylabel):
    plt.xlabel('')
    plt.ylabel(ylabel, fontproperties=prop, fontsize=22)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(18)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title('Task')
        legend.set_loc('lower left')
        plt.setp(legend.get_title(), fontproperties=prop, fontsize=20)
        plt.setp(legend.get_texts(), fontproperties=prop, fontsize=18)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['bottom'].set_linewidth(1.6)
    ax.tick_params(axis='both', width=1.6, length=6)
    sns.despine()
    plt.tight_layout()


# ======================================================================================================================
# E1 plots by task and condition
# ======================================================================================================================
group_cols = ['Subnum', 'Condition', 'Task']
exploration_trial_metrics = ['rank_2', 'EV_rank', 'EV_history']


def summarize_metric(metric):
    metric_df = E1_dm_switch.copy()
    if metric in exploration_trial_metrics:
        metric_df = metric_df[metric_df['exploration'] == 1].copy()

    metric_summary = (
        metric_df.groupby(group_cols, observed=False)[metric]
        .mean()
        .dropna()
        .reset_index()
    )
    metric_summary['Condition'] = pd.Categorical(
        metric_summary['Condition'],
        categories=condition_order,
        ordered=True,
    )
    metric_summary['Task'] = metric_summary['Task'].map({1: 'First', 2: 'Second'})
    metric_summary['Task'] = pd.Categorical(metric_summary['Task'], categories=task_order, ordered=True)
    return metric_summary


def summarize_ev_history_emmeans():
    coefficient_summary = EV_history_emmeans[['Condition', 'Task', 'emmean', 'SE']].copy()
    coefficient_summary['Condition'] = pd.Categorical(
        coefficient_summary['Condition'],
        categories=condition_order,
        ordered=True,
    )
    coefficient_summary['Task'] = pd.Categorical(
        coefficient_summary['Task'],
        categories=task_order,
        ordered=True,
    )
    return coefficient_summary.rename(columns={'emmean': 'mean', 'SE': 'se'})


def summarize_rank_2_emmeans():
    coefficient_summary = rank_2_emmeans[['Condition', 'Task', 'emmean', 'SE']].copy()
    coefficient_summary['Condition'] = pd.Categorical(
        coefficient_summary['Condition'],
        categories=condition_order,
        ordered=True,
    )
    coefficient_summary['Task'] = pd.Categorical(
        coefficient_summary['Task'],
        categories=task_order,
        ordered=True,
    )
    coefficient_summary['mean'] = 1 / (1 + np.exp(-coefficient_summary['emmean']))
    coefficient_summary['se'] = (
        coefficient_summary['mean']
        * (1 - coefficient_summary['mean'])
        * coefficient_summary['SE']
    )
    return coefficient_summary


def prepare_model_parameter_df():
    parameter_df = E1_model_params.copy()
    parameter_df['Condition'] = pd.Categorical(
        parameter_df['Condition'],
        categories=condition_order,
        ordered=True,
    )
    parameter_df['Task'] = parameter_df['Task'].replace({1: 'First', 2: 'Second'})
    parameter_df['Task'] = pd.Categorical(parameter_df['Task'], categories=task_order, ordered=True)
    return parameter_df


def summarize_model_parameter(metric):
    parameter_df = prepare_model_parameter_df()
    participant_summary = (
        parameter_df.groupby(['Subnum', 'Condition', 'Task'], observed=True)[metric]
        .mean()
        .dropna()
        .reset_index()
    )
    coefficient_summary = (
        participant_summary.groupby(['Condition', 'Task'], observed=True)[metric]
        .agg(['mean', 'sem'])
        .reset_index()
    )
    coefficient_summary['se'] = coefficient_summary['sem'].fillna(0)
    return coefficient_summary


e1_metrics = {
    'Reward': ('Reward', 'E1_Reward_by_Task_and_Condition.png'),
    'BestChoice': ('P(Optimal Choice)', 'E1_BestChoice_by_Task_and_Condition.png'),
    'value_gap': (r'$\Delta$ Best-Chosen Value', 'E1_Value_Gap_by_Task_and_Condition.png'),
    'Switch': ('P(Switch)', 'E1_Switch_by_Task_and_Condition.png'),
    'WinStay': ('P(Win-Stay)', 'E1_WinStay_by_Task_and_Condition.png'),
    'LoseShift': ('P(Lose-Shift)', 'E1_LoseShift_by_Task_and_Condition.png'),
    'exploration': ('P(Exploration)', 'E1_Exploration_by_Task_and_Condition.png'),
    'rank_2': ('P(Exploratory Second-Best Choice)', 'E1_Rank_2_by_Task_and_Condition.png'),
    'EV_history': ('Exploratory EV Chosen', 'E1_EV_History_by_Task_and_Condition.png'),
    't': (model_parameter_labels['t'], 'E1_Inverse_Temperature_by_Task_and_Condition.png'),
    'dis_sd': (model_parameter_labels['dis_sd'], 'E1_Reward_Variance_by_Task_and_Condition.png'),
    'noise_sd': (model_parameter_labels['noise_sd'], 'E1_Noise_Variance_by_Task_and_Condition.png'),
    'decay': (model_parameter_labels['decay'], 'E1_Decay_Rate_by_Task_and_Condition.png'),
    'decay_center': (model_parameter_labels['decay_center'], 'E1_Decay_Center_by_Task_and_Condition.png'),
}

reference_lines = {
    'Reward': 50,
}

for metric, (ylabel, filename) in e1_metrics.items():
    if metric == 'EV_history':
        coefficient_summary = summarize_ev_history_emmeans()
    elif metric == 'rank_2':
        coefficient_summary = summarize_rank_2_emmeans()
    elif metric in model_parameter_cols:
        coefficient_summary = summarize_model_parameter(metric)
    else:
        metric_summary = summarize_metric(metric)
        coefficient_summary = (
            metric_summary.groupby(['Condition', 'Task'], observed=True)[metric]
            .agg(['mean', 'sem'])
            .reset_index()
        )
        coefficient_summary['se'] = coefficient_summary['sem'].fillna(0)

    fig, ax = plt.subplots(figsize=(5, 6.5))
    x_positions = {condition: idx for idx, condition in enumerate(condition_order)}
    task_offsets = {'First': -0.16, 'Second': 0.16}

    for _, row in coefficient_summary.iterrows():
        task = row['Task']
        condition = row['Condition']
        x = x_positions[condition] + task_offsets[task]
        ax.errorbar(
            x,
            row['mean'],
            yerr=row['se'],
            fmt='o',
            markersize=12,
            color='black',
            ecolor='black',
            elinewidth=2.5,
            capsize=5,
            markerfacecolor=task_palette[task],
            markeredgecolor='black',
            markeredgewidth=2,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            linestyle='',
            markerfacecolor=task_palette[task],
            markeredgecolor='black',
            markeredgewidth=1.5,
            color='black',
            markersize=11,
            label=task,
        )
        for task in task_order
    ]
    ax.legend(handles=handles, title='Task', loc='lower left')
    reference_line = reference_lines.get(metric, 0)
    ax.axhline(reference_line, color='gray', linestyle='--', linewidth=1.8)
    ax.set_xticks(np.arange(len(condition_order)))
    ax.set_xticklabels(condition_order)
    ax.set_xlim(-0.5, len(condition_order) - 0.5)
    apply_e1_plot_style(ax, ylabel)
    coefficient_filename = filename.replace('_by_Task_and_Condition.png', '_coefficient_by_Task_and_Condition.png')
    plt.savefig(f'./figures/{coefficient_filename}', dpi=600)
    plt.close()


# ======================================================================================================================
# E2 plots by condition
# ======================================================================================================================
e2_group_cols = ['Subnum', 'Condition']


def summarize_e2_metric(metric):
    metric_df = E2_dm_switch.copy()
    if metric in exploration_trial_metrics:
        metric_df = metric_df[metric_df['exploration'] == 1].copy()

    participant_summary = (
        metric_df.groupby(e2_group_cols, observed=False)[metric]
        .mean()
        .dropna()
        .reset_index()
    )
    participant_summary['Condition'] = pd.Categorical(
        participant_summary['Condition'],
        categories=condition_order,
        ordered=True,
    )
    coefficient_summary = (
        participant_summary.groupby('Condition', observed=True)[metric]
        .agg(['mean', 'sem'])
        .reset_index()
    )
    coefficient_summary['se'] = coefficient_summary['sem'].fillna(0)
    return coefficient_summary


def summarize_e2_ev_history_emmeans():
    coefficient_summary = E2_EV_history_emmeans[['Condition', 'emmean', 'SE']].copy()
    coefficient_summary['Condition'] = pd.Categorical(
        coefficient_summary['Condition'],
        categories=condition_order,
        ordered=True,
    )
    return coefficient_summary.rename(columns={'emmean': 'mean', 'SE': 'se'})


def summarize_e2_rank_2_emmeans():
    coefficient_summary = E2_rank_2_emmeans[['Condition', 'emmean', 'SE']].copy()
    coefficient_summary['Condition'] = pd.Categorical(
        coefficient_summary['Condition'],
        categories=condition_order,
        ordered=True,
    )
    coefficient_summary['mean'] = 1 / (1 + np.exp(-coefficient_summary['emmean']))
    coefficient_summary['se'] = (
        coefficient_summary['mean']
        * (1 - coefficient_summary['mean'])
        * coefficient_summary['SE']
    )
    return coefficient_summary


def summarize_e2_model_parameter(metric):
    parameter_df = E2_model_params.copy()
    parameter_df['Condition'] = pd.Categorical(
        parameter_df['Condition'],
        categories=condition_order,
        ordered=True,
    )
    participant_summary = (
        parameter_df.groupby(e2_group_cols, observed=True)[metric]
        .mean()
        .dropna()
        .reset_index()
    )
    coefficient_summary = (
        participant_summary.groupby('Condition', observed=True)[metric]
        .agg(['mean', 'sem'])
        .reset_index()
    )
    coefficient_summary['se'] = coefficient_summary['sem'].fillna(0)
    return coefficient_summary


e2_metrics = {
    'Reward': ('Reward', 'E2_Reward_coefficient_by_Condition.png'),
    'BestChoice': ('P(Optimal Choice)', 'E2_BestChoice_coefficient_by_Condition.png'),
    'value_gap': (r'$\Delta$ Best-Chosen Value', 'E2_Value_Gap_coefficient_by_Condition.png'),
    'Switch': ('P(Switch)', 'E2_Switch_coefficient_by_Condition.png'),
    'WinStay': ('P(Win-Stay)', 'E2_WinStay_coefficient_by_Condition.png'),
    'LoseShift': ('P(Lose-Shift)', 'E2_LoseShift_coefficient_by_Condition.png'),
    'exploration': ('P(Exploration)', 'E2_Exploration_coefficient_by_Condition.png'),
    'rank_2': ('P(Exploratory Second-Best Choice)', 'E2_Rank_2_coefficient_by_Condition.png'),
    'EV_history': ('Exploratory EV Chosen', 'E2_EV_History_coefficient_by_Condition.png'),
    't': (model_parameter_labels['t'], 'E2_Inverse_Temperature_coefficient_by_Condition.png'),
    'dis_sd': (model_parameter_labels['dis_sd'], 'E2_Reward_Variance_coefficient_by_Condition.png'),
    'noise_sd': (model_parameter_labels['noise_sd'], 'E2_Noise_Variance_coefficient_by_Condition.png'),
    'decay': (model_parameter_labels['decay'], 'E2_Decay_Rate_coefficient_by_Condition.png'),
    'decay_center': (model_parameter_labels['decay_center'], 'E2_Decay_Center_coefficient_by_Condition.png'),
}

for metric, (ylabel, filename) in e2_metrics.items():
    if metric == 'EV_history':
        coefficient_summary = summarize_e2_ev_history_emmeans()
    elif metric == 'rank_2':
        coefficient_summary = summarize_e2_rank_2_emmeans()
    elif metric in model_parameter_cols:
        coefficient_summary = summarize_e2_model_parameter(metric)
    else:
        coefficient_summary = summarize_e2_metric(metric)

    fig, ax = plt.subplots(figsize=(3.8, 6.5))
    x_positions = {condition: idx for idx, condition in enumerate(condition_order)}

    for _, row in coefficient_summary.iterrows():
        condition = row['Condition']
        x = x_positions[condition]
        ax.errorbar(
            x,
            row['mean'],
            yerr=row['se'],
            fmt='o',
            markersize=12,
            color='black',
            ecolor='black',
            elinewidth=2.5,
            capsize=5,
            markerfacecolor=condition_palette[condition],
            markeredgecolor='black',
            markeredgewidth=2,
        )

    reference_line = reference_lines.get(metric, 0)
    ax.axhline(reference_line, color='gray', linestyle='--', linewidth=1.8)
    ax.set_xticks(np.arange(len(condition_order)))
    ax.set_xticklabels(condition_order)
    ax.set_xlim(-0.45, len(condition_order) - 0.55)
    apply_e1_plot_style(ax, ylabel)
    plt.savefig(f'./figures/{filename}', dpi=600)
    plt.close()

# ======================================================================================================================
# Reward by exploration status
# ======================================================================================================================
reward_strategy_summary = (
    E1_dm_switch.dropna(subset=['exploration'])
    .groupby(['Subnum', 'Condition', 'Task', 'exploration'], observed=False)['Reward']
    .mean()
    .dropna()
    .reset_index()
)
reward_strategy_summary['Condition'] = pd.Categorical(
    reward_strategy_summary['Condition'],
    categories=condition_order,
    ordered=True,
)
reward_strategy_summary['Task'] = reward_strategy_summary['Task'].map({1: 'First', 2: 'Second'})
reward_strategy_summary['Task'] = pd.Categorical(reward_strategy_summary['Task'], categories=task_order, ordered=True)
reward_strategy_summary['Strategy'] = reward_strategy_summary['exploration'].map({
    0: 'Exploitation',
    1: 'Exploration',
})
reward_strategy_summary['Strategy'] = pd.Categorical(
    reward_strategy_summary['Strategy'],
    categories=strategy_order,
    ordered=True,
)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.5), sharey=True)
x_positions = {condition: idx for idx, condition in enumerate(condition_order)}
strategy_offsets = {'Exploitation': -0.16, 'Exploration': 0.16}

for ax, task in zip(axes, task_order):
    task_df = reward_strategy_summary[reward_strategy_summary['Task'] == task]
    task_summary = (
        task_df.groupby(['Condition', 'Strategy'], observed=True)['Reward']
        .agg(['mean', 'sem'])
        .reset_index()
    )
    task_summary['se'] = task_summary['sem'].fillna(0)

    for _, row in task_summary.iterrows():
        condition = row['Condition']
        strategy = row['Strategy']
        x = x_positions[condition] + strategy_offsets[strategy]
        ax.errorbar(
            x,
            row['mean'],
            yerr=row['se'],
            fmt='o',
            markersize=12,
            color='black',
            ecolor='black',
            elinewidth=2.5,
            capsize=5,
            markerfacecolor=strategy_palette[strategy],
            markeredgecolor='black',
            markeredgewidth=2,
        )

    ax.axhline(50, color='gray', linestyle='--', linewidth=1.8)
    ax.set_title(f'{task} Task', fontproperties=prop, fontsize=24)
    ax.set_xticks(np.arange(len(condition_order)))
    ax.set_xticklabels(condition_order)
    ax.set_xlim(-0.5, len(condition_order) - 0.5)
    ax.set_xlabel('')
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(18)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['bottom'].set_linewidth(1.6)
    ax.tick_params(axis='both', width=1.6, length=6)

axes[0].set_ylabel('Reward', fontproperties=prop, fontsize=22)
axes[1].set_ylabel('')
handles = [
    plt.Line2D(
        [0],
        [0],
        marker='o',
        linestyle='',
        markerfacecolor=strategy_palette[strategy],
        markeredgecolor='black',
        markeredgewidth=1.5,
        color='black',
        markersize=11,
        label=strategy,
    )
    for strategy in strategy_order
]
axes[0].legend(handles=handles, title='Choice Type')
legend = axes[0].get_legend()
plt.setp(legend.get_title(), fontproperties=prop, fontsize=20)
plt.setp(legend.get_texts(), fontproperties=prop, fontsize=18)
sns.despine()
plt.tight_layout()
plt.savefig('./figures/E1_Reward_by_Exploration_Status_Task_Condition.png', dpi=600)
plt.close()

# ======================================================================================================================
# Best-fitting model parameter table
# ======================================================================================================================
model_param_df = prepare_model_parameter_df()

model_param_summary = (
    model_param_df.groupby(['Condition', 'Task'], observed=True)
    .agg(
        Model=('Model', lambda x: ', '.join(sorted(x.dropna().unique()))),
        N=('Subnum', 'nunique'),
        **{
            f'{param}_mean': (param, 'mean')
            for param in model_parameter_cols
        },
        **{
            f'{param}_sd': (param, 'std')
            for param in model_parameter_cols
        },
    )
    .reset_index()
)

model_param_table = model_param_summary[['Condition', 'Task', 'Model', 'N']].copy()
for param in model_parameter_cols:
    mean_col = f'{param}_mean'
    sd_col = f'{param}_sd'
    model_param_table[model_parameter_labels[param]] = model_param_summary.apply(
        lambda row: f'{row[mean_col]:.2f} +/- {row[sd_col]:.2f}',
        axis=1,
    )

try:
    _write_model_parameter_docx(model_param_table, model_parameter_docx_path)
except PermissionError:
    docx_path = Path(model_parameter_docx_path)
    fallback_docx_path = docx_path.with_name(f'{docx_path.stem}_plain{docx_path.suffix}')
    _write_model_parameter_docx(model_param_table, fallback_docx_path)
    print(f'Could not overwrite {model_parameter_docx_path}; wrote {fallback_docx_path} instead.')
