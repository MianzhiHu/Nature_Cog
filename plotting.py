import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib import colors as mpl_colors
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape as xml_escape
from robustness_check import E2_sig, E1_trial_rating_all_sig

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})


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
E2_semantic_coefficients = pd.read_csv('./data/e2_semantic_feature_model_coefficients_R.csv')
E1_trial_rating_coefficients = pd.read_csv('./data/e1_trialwise_semantic_rating_model_coefficients_R.csv')


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
# E2 significant semantic-feature effects
# ======================================================================================================================
def apply_e2_semantic_plot_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontproperties=prop, fontsize=22)
    ax.set_ylabel(ylabel, fontproperties=prop, fontsize=22)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(18)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(18)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title('Condition')
        legend.set_loc('lower right')
        plt.setp(legend.get_title(), fontproperties=prop, fontsize=20)
        plt.setp(legend.get_texts(), fontproperties=prop, fontsize=18)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['bottom'].set_linewidth(1.6)
    ax.tick_params(axis='both', width=1.6, length=6)
    sns.despine()
    plt.tight_layout()


E2_sig_plot_effects = E2_sig.drop_duplicates(
    subset=['level', 'outcome', 'feature', 'model_type', 'term']
).copy()

for _, effect in E2_sig_plot_effects.iterrows():
    outcome = effect['outcome']
    feature = effect['feature']
    model_type = effect['model_type']

    plot_data = E2_dm_switch.loc[
        E2_dm_switch['Condition'].isin(['Nature', 'Urban']),
        ['Condition', feature, outcome],
    ].dropna().copy()

    feature_reference = E2_dm_switch.loc[
        E2_dm_switch['Condition'].isin(['Nature', 'Urban']),
        feature,
    ].dropna()
    feature_mean = feature_reference.mean()
    feature_sd = feature_reference.std()
    plot_data['feature_z'] = (plot_data[feature] - feature_mean) / feature_sd

    observed_summary = (
        plot_data.groupby(['Condition', 'feature_z'], observed=True)[outcome]
        .mean()
        .reset_index()
    )

    model_coefficients = E2_semantic_coefficients[
        (E2_semantic_coefficients['level'] == effect['level'])
        & (E2_semantic_coefficients['outcome'] == outcome)
        & (E2_semantic_coefficients['feature'] == feature)
        & (E2_semantic_coefficients['model_type'] == model_type)
    ].set_index('term')['Estimate']

    intercept = model_coefficients['(Intercept)']
    feature_effect = model_coefficients['feature_z']
    condition_effect = model_coefficients.get('ConditionUrban', 0)
    interaction_effect = model_coefficients.get('feature_z:ConditionUrban', 0)

    fig, ax = plt.subplots(figsize=(5, 6.5))

    for condition in ['Nature', 'Urban']:
        condition_data = observed_summary[observed_summary['Condition'] == condition]
        ax.scatter(
            condition_data['feature_z'],
            condition_data[outcome],
            s=45,
            color=condition_palette[condition],
            edgecolor='black',
            linewidth=0.7,
            alpha=0.55,
        )

        x_grid = np.linspace(
            condition_data['feature_z'].min(),
            condition_data['feature_z'].max(),
            200,
        )
        linear_predictor = intercept + feature_effect * x_grid
        if condition == 'Urban':
            linear_predictor += condition_effect + interaction_effect * x_grid

        if effect['family'] == 'binomial':
            predicted_outcome = 1 / (1 + np.exp(-np.clip(linear_predictor, -700, 700)))
        else:
            predicted_outcome = linear_predictor

        ax.plot(
            x_grid,
            predicted_outcome,
            color=condition_palette[condition],
            linewidth=3,
            label=condition,
        )

    ax.legend(title='Condition', loc='lower right')

    if effect['family'] == 'binomial':
        ax.set_ylim(-0.03, 1.03)

    effect_name = 'Interaction' if model_type == 'interaction' else 'Feature'
    adjusted_p = effect['p_value_adjusted']
    adjusted_p_text = f'{adjusted_p:.3f}' if adjusted_p >= 0.001 else '< 0.001'
    adjusted_p_operator = '=' if adjusted_p >= 0.001 else ''
    ax.text(
        0.03,
        0.97,
        f'{effect_name} FDR-adjusted p {adjusted_p_operator}{adjusted_p_text}',
        transform=ax.transAxes,
        va='top',
        fontproperties=prop,
        fontsize=16,
    )

    ylabel = e2_metrics.get(outcome, (outcome, ''))[0]
    xlabel = f'{feature.replace("_", " ").title()} (standardized)'
    apply_e2_semantic_plot_style(ax, xlabel, ylabel)

    output_name = f'E2_{outcome}_{feature}_{model_type}_semantic_effect.png'
    plt.savefig(f'./figures/{output_name}', dpi=600)
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

# ======================================================================================================================
# E2 significant semantic-feature effects: raw trial-level plots
# ======================================================================================================================
# Each scatter point below is one unaveraged E2 trial from the same Nature-versus-Urban data used in the mixed models.
# Feature values of zero are retained because feature absence was included when the models were fitted.
# Small deterministic jitter is used only to reveal overlapping points; model predictions use the unjittered values.
# This final block overwrites the earlier binned PNGs with raw-data versions and intentionally exports PNG only.
raw_plot_rng = np.random.default_rng(20260804)

for _, effect in E2_sig_plot_effects.iterrows():
    outcome = effect['outcome']
    feature = effect['feature']
    model_type = effect['model_type']

    raw_plot_data = E2_dm_switch.loc[
        E2_dm_switch['Condition'].isin(['Nature', 'Urban']),
        ['Subnum', 'Condition', feature, outcome],
    ].dropna().copy()

    # Match the feature standardization used in the R models before outcome-specific missing rows were removed.
    feature_reference = E2_dm_switch.loc[
        E2_dm_switch['Condition'].isin(['Nature', 'Urban']),
        feature,
    ].dropna()
    feature_mean = feature_reference.mean()
    feature_sd = feature_reference.std()
    raw_plot_data['feature_z'] = (raw_plot_data[feature] - feature_mean) / feature_sd

    model_coefficients = E2_semantic_coefficients[
        (E2_semantic_coefficients['level'] == effect['level'])
        & (E2_semantic_coefficients['outcome'] == outcome)
        & (E2_semantic_coefficients['feature'] == feature)
        & (E2_semantic_coefficients['model_type'] == model_type)
    ].set_index('term')['Estimate']

    intercept = model_coefficients['(Intercept)']
    feature_effect = model_coefficients['feature_z']
    condition_effect = model_coefficients.get('ConditionUrban', 0)
    interaction_effect = model_coefficients.get('feature_z:ConditionUrban', 0)

    fig, ax = plt.subplots(figsize=(5, 6.5))

    for condition in ['Nature', 'Urban']:
        condition_data = raw_plot_data[raw_plot_data['Condition'] == condition]

        # Jitter affects display positions only and does not alter the raw data or fitted model lines.
        x_display = condition_data['feature_z'].to_numpy() + raw_plot_rng.normal(
            loc=0,
            scale=0.015,
            size=len(condition_data),
        )
        y_display = condition_data[outcome].to_numpy().copy()
        if effect['family'] == 'binomial':
            y_display = y_display + raw_plot_rng.normal(
                loc=0,
                scale=0.018,
                size=len(condition_data),
            )

        ax.scatter(
            x_display,
            y_display,
            s=12,
            color=condition_palette[condition],
            edgecolor='none',
            alpha=0.09,
            rasterized=True,
        )

        # Draw the fixed-effect prediction only across the feature range observed in this condition.
        x_grid = np.linspace(
            condition_data['feature_z'].min(),
            condition_data['feature_z'].max(),
            200,
        )
        linear_predictor = intercept + feature_effect * x_grid
        if condition == 'Urban':
            linear_predictor += condition_effect + interaction_effect * x_grid

        if effect['family'] == 'binomial':
            predicted_outcome = 1 / (1 + np.exp(-np.clip(linear_predictor, -700, 700)))
        else:
            predicted_outcome = linear_predictor

        ax.plot(
            x_grid,
            predicted_outcome,
            color=condition_palette[condition],
            linewidth=3,
            label=condition,
        )

    ax.legend(title='Condition', loc='lower right')
    if effect['family'] == 'binomial':
        ax.set_ylim(-0.07, 1.07)

    effect_name = 'Interaction' if model_type == 'interaction' else 'Feature'
    adjusted_p = effect['p_value_adjusted']
    adjusted_p_text = f'{adjusted_p:.3f}' if adjusted_p >= 0.001 else '< 0.001'
    adjusted_p_operator = '=' if adjusted_p >= 0.001 else ''
    ax.text(
        0.03,
        0.97,
        f'{effect_name} FDR-adjusted p {adjusted_p_operator}{adjusted_p_text}',
        transform=ax.transAxes,
        va='top',
        fontproperties=prop,
        fontsize=16,
    )

    ylabel = e2_metrics.get(outcome, (outcome, ''))[0]
    xlabel = f'{feature.replace("_", " ").title()} (standardized)'
    apply_e2_semantic_plot_style(ax, xlabel, ylabel)

    output_name = f'E2_{outcome}_{feature}_{model_type}_semantic_effect.png'
    plt.savefig(f'./figures/{output_name}', dpi=600)
    plt.close()

# ======================================================================================================================
# E1 trial-wise semantic effects surviving the pooled global FDR correction
# ======================================================================================================================
# One asymmetric figure replaces 15 separate panels: additive effects are combined by feature and rating, while the
# naturalness interactions share one coefficient axis. Open interaction markers flag weak Nature-image support (<10).
e1_global_sig = E1_trial_rating_all_sig.copy()
e1_additive_sig = e1_global_sig[e1_global_sig['model_type'] == 'additive'].copy()
e1_interaction_sig = e1_global_sig[e1_global_sig['model_type'] == 'interaction'].copy()

e1_additive_sig['semantic_feature'] = e1_additive_sig['term'].str.replace('_z', '', regex=False)
e1_interaction_sig['semantic_feature'] = (
    e1_interaction_sig['term'].str.replace('_z:ConditionUrban', '', regex=False)
)
e1_additive_sig['p_marker_size'] = np.select(
    [
        e1_additive_sig['p_value_adjusted_all'] < 0.001,
        e1_additive_sig['p_value_adjusted_all'] < 0.01,
        e1_additive_sig['p_value_adjusted_all'] < 0.05,
    ],
    [840, 720, 600],
    default=np.nan,
)

feature_order = ['road', 'fence', 'grass', 'mountain']
rating_order = ['aesthetic', 'familiarity', 'engagement', 'fascination', 'mystery', 'imagability', 'control']
rating_labels = {
    'aesthetic': 'Aesthetic',
    'familiarity': 'Familiarity',
    'engagement': 'Engagement',
    'fascination': 'Fascination',
    'mystery': 'Mystery',
    'imagability': 'Imageability',
    'control': 'Control',
}
x_positions = {rating: index for index, rating in enumerate(rating_order)}
y_positions = {feature: len(feature_order) - index - 1 for index, feature in enumerate(feature_order)}

# Separate figure 1: additive feature-by-rating associations.
fig_additive, ax_effect_map = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)

max_abs_additive = e1_additive_sig['Estimate'].abs().max()
effect_norm = mpl_colors.TwoSlopeNorm(vmin=-max_abs_additive, vcenter=0, vmax=max_abs_additive)
effect_cmap = plt.get_cmap('RdBu_r')

for _, effect in e1_additive_sig.iterrows():
    x_value = x_positions[effect['outcome']]
    y_value = y_positions[effect['semantic_feature']]
    marker_size = effect['p_marker_size']
    marker_color = effect_cmap(effect_norm(effect['Estimate']))
    luminance = 0.299 * marker_color[0] + 0.587 * marker_color[1] + 0.114 * marker_color[2]
    label_color = 'white' if luminance < 0.55 else '#272727'

    ax_effect_map.scatter(
        x_value, y_value, s=marker_size, color=marker_color,
        edgecolor='white', linewidth=0.8, zorder=3,
    )
    ax_effect_map.text(
        x_value, y_value, f"{effect['Estimate']:.2f}",
        ha='center', va='center', fontproperties=prop, fontsize=12, color=label_color, zorder=4,
    )

ax_effect_map.set_xticks(range(len(rating_order)))
ax_effect_map.set_xticklabels(
    [rating_labels[rating] for rating in rating_order],
    rotation=30, ha='right', rotation_mode='anchor',
)
ax_effect_map.set_yticks(range(len(feature_order)))
ax_effect_map.set_yticklabels([feature.title() for feature in reversed(feature_order)])
for tick_label in ax_effect_map.get_xticklabels() + ax_effect_map.get_yticklabels():
    tick_label.set_fontproperties(prop)
    tick_label.set_fontsize(12)
ax_effect_map.set_xlim(-0.55, len(rating_order) - 0.45)
ax_effect_map.set_ylim(-0.55, len(feature_order) - 0.45)
ax_effect_map.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
ax_effect_map.tick_params(axis='both', length=0)

p_legend_labels = ['p < .001', 'p < .01', 'p < .05']
p_legend_sizes = [720, 600, 480]
p_legend_handles = [
    ax_effect_map.scatter(
        [], [], s=marker_size,
        facecolor='#B4C0E4', edgecolor='white', linewidth=0.8,
    )
    for marker_size in p_legend_sizes
]
additive_legend = ax_effect_map.legend(
    p_legend_handles, p_legend_labels,
    title= 'p adjusted fdr', bbox_to_anchor=(1.2, 0.5),
    borderaxespad=0, handletextpad=1.0, handleheight=2.5, labelspacing=1.5,
    fontsize=10, title_fontsize=12,
)
plt.setp(additive_legend.get_title(), fontproperties=prop, fontsize=12)
plt.setp(additive_legend.get_texts(), fontproperties=prop, fontsize=10)
coefficient_mappable = plt.cm.ScalarMappable(norm=effect_norm, cmap=effect_cmap)
coefficient_colorbar = fig_additive.colorbar(
    coefficient_mappable, ax=ax_effect_map, orientation='horizontal',
    fraction=0.06, pad=0.05, aspect=35,
)
coefficient_colorbar.set_label(
    'Fixed-effect coefficient',
    fontproperties=prop, fontsize=12, labelpad=2,
)
coefficient_colorbar.outline.set_visible(False)
coefficient_colorbar.ax.tick_params(length=2, labelsize=10)
for tick_label in coefficient_colorbar.ax.get_xticklabels():
    tick_label.set_fontproperties(prop)
    tick_label.set_fontsize(10)

additive_output_stem = './figures/E1_trialwise_semantic_rating_global_fdr_additive_effects'
fig_additive.savefig(f'{additive_output_stem}.png', dpi=600, bbox_inches='tight')
plt.close(fig_additive)




# Combined figures 2 and 3: interaction estimates and condition-specific slopes share the feature axis.
interaction_order = ['building', 'railing', 'grass', 'signboard']
interaction_y = {
    feature: len(interaction_order) - index - 1
    for index, feature in enumerate(interaction_order)
}
interaction_ticks = [0, 1, 2, 10, 100]
slope_change_ticks = [-100, -20, -10, -2, -1, 0, 1, 2]

# Nature is the reference condition; the Urban slope is the Nature slope plus the Urban interaction term.
naturalness_interaction_coefficients = E1_trial_rating_coefficients[
    (E1_trial_rating_coefficients['outcome'] == 'naturalness')
    & (E1_trial_rating_coefficients['model_type'] == 'interaction')
    & (E1_trial_rating_coefficients['converged'] == True)
    & (E1_trial_rating_coefficients['singular'] == False)
].set_index('term')

condition_slope_rows = []
for feature in interaction_order:
    nature_slope = naturalness_interaction_coefficients.loc[f'{feature}_z', 'Estimate']
    urban_slope = nature_slope + naturalness_interaction_coefficients.loc[
        f'{feature}_z:ConditionUrban', 'Estimate'
    ]
    condition_slope_rows.append({
        'feature': feature,
        'nature_slope': nature_slope,
        'urban_slope': urban_slope,
    })
condition_slopes = pd.DataFrame(condition_slope_rows).set_index('feature')

fig_combined, (ax_interaction_panel, ax_slope_panel) = plt.subplots(
    1, 2, figsize=(8, 5), sharey=True, constrained_layout=True,
    gridspec_kw={'width_ratios': [1, 1.10], 'wspace': 0.10},
)

for _, effect in e1_interaction_sig.iterrows():
    feature = effect['semantic_feature']
    y_value = interaction_y[feature]
    estimate = effect['Estimate']
    ci_half_width = 1.96 * effect['Std. Error']
    ax_interaction_panel.errorbar(
        estimate, y_value, xerr=ci_half_width, fmt='o', markersize=10,
        markerfacecolor='#B64342', markeredgecolor='#B64342', markeredgewidth=2,
        ecolor='#B64342', elinewidth=2, capsize=5, zorder=3
    )

ax_interaction_panel.axvline(0, color='#767676', linewidth=1.2, linestyle='--', zorder=1)
ax_interaction_panel.set_xscale('symlog', linthresh=2, linscale=0.9, base=10)
ax_interaction_panel.set_xlim(-0.12, 140)
ax_interaction_panel.set_xticks(interaction_ticks)
ax_interaction_panel.set_xticklabels([f'{tick:g}' for tick in interaction_ticks])
ax_interaction_panel.set_yticks(range(len(interaction_order)))
ax_interaction_panel.set_yticklabels([feature.title() for feature in reversed(interaction_order)])
ax_interaction_panel.set_ylim(-0.55, len(interaction_order) - 0.45)
ax_interaction_panel.set_xlabel(
    'Interaction Effects (Urban \N{MINUS SIGN} Nature)',
    fontproperties=prop, fontsize=12,
)
ax_interaction_panel.spines[['top', 'right', 'left']].set_visible(False)
ax_interaction_panel.tick_params(axis='y', length=0)
ax_interaction_panel.tick_params(axis='x', length=2)

for feature in interaction_order:
    y_value = interaction_y[feature]
    nature_slope = condition_slopes.loc[feature, 'nature_slope']
    urban_slope = condition_slopes.loc[feature, 'urban_slope']
    ax_slope_panel.annotate(
        '', xy=(urban_slope, y_value), xytext=(nature_slope, y_value),
        arrowprops={
            'arrowstyle': '->', 'color': '#9A9A9A', 'linewidth': 2,
            'mutation_scale': 10, 'shrinkA': 6, 'shrinkB': 6,
        },
        zorder=1,
    )
    ax_slope_panel.scatter(
        nature_slope, y_value,
        s=200, color=nature_color, edgecolor='white', linewidth=1.2, zorder=3,
    )
    ax_slope_panel.scatter(
        urban_slope, y_value,
        s=200, color=urban_color, edgecolor='white', linewidth=1.2, zorder=3,
    )

ax_slope_panel.axvline(0, color='#767676', linewidth=1.2, linestyle='--', zorder=0)
ax_slope_panel.set_xscale('symlog', linthresh=2, linscale=0.9, base=10)
ax_slope_panel.set_xlim(-120, 3)
ax_slope_panel.set_xticks(slope_change_ticks)
ax_slope_panel.set_xticklabels([f'{tick:g}' for tick in slope_change_ticks])
ax_slope_panel.set_xlabel(
    'Condition-Specific Feature Slope with Naturalness',
    fontproperties=prop, fontsize=12,
)
ax_slope_panel.spines[['top', 'right', 'left']].set_visible(False)
ax_slope_panel.tick_params(axis='y', left=False, labelleft=False)
ax_slope_panel.tick_params(axis='x', length=2)

for axis in (ax_interaction_panel, ax_slope_panel):
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontproperties(prop)
        tick_label.set_fontsize(12)

combined_nature_handle = ax_slope_panel.scatter(
    [], [], s=200, color=nature_color, edgecolor='white', linewidth=1.2,
)
combined_urban_handle = ax_slope_panel.scatter(
    [], [], s=200, color=urban_color, edgecolor='white', linewidth=1.2,
)
# combined_legend = ax_slope_panel.legend(
#     [combined_nature_handle, combined_urban_handle], ['Nature', 'Urban'],
#     loc='lower center', bbox_to_anchor=(0.5, -0.28),
#     ncol=2, frameon=False, handletextpad=0.5, columnspacing=0.5,
# )
# plt.setp(combined_legend.get_texts(), fontproperties=prop, fontsize=10)

combined_output_stem = './figures/E1_trialwise_naturalness_interaction_and_slope_changes'
fig_combined.savefig(f'{combined_output_stem}.png', dpi=600, bbox_inches='tight')
plt.close(fig_combined)