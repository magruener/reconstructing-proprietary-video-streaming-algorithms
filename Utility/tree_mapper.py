import pickle

import numpy as np
import pydotplus
from matplotlib import cm
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

"""
Small tool to map a tree parsed from a file to a more readable version of the features
"""

feature_type_mapper = {  ## Map for features !!! needs to be adapted if the features are changed
    'future_buffer_filling': 'Buffer Fill / Download Time (s)',
    'future_mbit_discounted': '(Buffer Fill / Download Time (s)) x Mbit Gain',
    'future_vmaf_discounted': '(Buffer Fill / Download Time (s)) x VMAF Gain',
    'linear_qoe_normalized': 'Linear QoE Normalised',
    'linear_qoe_raw_switch': 'Linear QoE',
    'linear_qoe_normalized_summary_function_max': 'Linear QoE Normalised Best Improvement',
    'linear_qoe_raw_summary_function_max': 'Linear QoE Best Improvement',
    'future_download_time': 'Download time (s)',
    'throughput_variance': 'Throughput Variance',
    'relative_chunk_remain': '# Chunks to Play',
    'future_chunk_bitrate': 'Chunk Bitrate',
    'future_chunk_size_byte': 'Chunk Size (Byte)',
    'future_chunk_vmaf': 'Chunk VMAF',
    'timestamp_s': 'Time since Startup (s)',
    'download_time_s': 'Downloadtime (s)',
    'relative_rate_played': 'Rate Current / Rate Max',
    'data_used_bytes_relative': 'Bytes Downloads / Bytes needed for max Quality',
    'single_mbitrate': 'Mbit Rate',
    'encoded_mbitrate': 'Mbit Rate Average',
    'vmaf': 'VMAF',
    'segment_length_s': 'Segment Length (s)',
    'expected_segment_length_s': 'Expected Segment Length (s)',
    'video_chunk_size_byte': 'Size (Byte)',
    'buffer_size_s': 'Buffer (s)',
    'rebuffer_time_s': 'Rebuffer (s)',
    'sleep_time_s': 'Buffer Drainage (s)',
    'current_level': 'Quality Level'
}

same = '⇨'  ## Symbols for movements
up = '⇧'
low = '⇩'

reference_colormap = cm.get_cmap('winter', 256)  ## We're taking the most intese color of the map
MAX_N_SAMPLES_CONFIDENCE = 100  # How many samples do we need for full confidence in the decisison


def convert_rgba2hex(color):
    return '#{:02x}{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2], color[3])


def parse_tree(file_path, outputfile):
    """

    :param file_path: Filepath of the tree !! Use same python version as for training and saving
    :param outputfile : Where do we save the treefile we output
    :return:
    """
    with open(file_path, 'rb') as tree_file:
        tree_model = pickle.load(tree_file)
    feature_names = tree_model.extract_features_names()
    decision_model = tree_model.classifier
    dot_data = StringIO()
    export_graphviz(decision_model, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,
                    label='all',
                    class_names=decision_model.classes_.astype(str),
                    impurity=False,
                    feature_names=feature_names)

    input_label = ['class = %d' % i for i in range(tree_model.max_quality_change * 2 + 1)]
    output_label = ['Action : ' + ''.join([low] * i) for i in range(tree_model.max_quality_change, 0, -1)]
    output_label += ['Action : ' + same]
    output_label += ['Action : ' + ''.join([up] * i) for i in range(1, tree_model.max_quality_change + 1)]
    class_label_mapper = {in_l: out_l for in_l, out_l in zip(input_label, output_label)}

    input_label = ['_switch_%d' % i for i in range(-tree_model.max_quality_change, tree_model.max_quality_change + 1)]
    output_label = ['Switch : ' + ''.join([low] * i) for i in range(tree_model.max_quality_change, 0, -1)]
    output_label += ['Switch : ' + same]
    output_label += ['Switch : ' + ''.join([up] * i) for i in range(1, tree_model.max_quality_change + 1)]
    switch_mapper = [(in_l, out_l) for in_l, out_l in zip(input_label, output_label)]

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    for n in graph.get_nodes():
        label = n.get_label()
        if label:
            label_parsed = label[1:-1].split('<br/>')
            if len(label_parsed) == 3:
                sample_size, values, class_label = label[1:-1].split('<br/>')
            else:
                feature_name, sample_size, values, class_label = label[1:-1].split('<br/>')
            certainty = min(float(sample_size.split('=')[-1].strip()) / MAX_N_SAMPLES_CONFIDENCE, 1.0)
            label_parsed = parse_node_label(label, class_label_mapper, switch_mapper)
            n.set_label(label_parsed)
            reference_color = np.array(reference_colormap(1.0))
            reference_color[-1] = certainty
            reference_color *= reference_color
            n.set_fillcolor(convert_rgba2hex((reference_color * 255).astype(int)))
    png_binary = graph.create_png()
    with open(outputfile, 'wb') as outpng:
        outpng.write(png_binary)


def parse_node_label(label, class_label_mapper, switch_mapper):
    label_parsed = label[1:-1].split('<br/>')
    if len(label_parsed) == 3:
        sample_size, values, class_label = label[1:-1].split('<br/>')
        return class_label_mapper[class_label]
    feature_name, sample_size, values, class_label = label[1:-1].split('<br/>')
    feature, value = feature_name.split('&')
    value = 'value &' + value
    feature = feature.strip()
    try:
        if 'lookback' not in feature and 'future' not in feature:
            feature_lookback = str(-int(feature.split('_')[-1]))
            feature = '_'.join(feature.split('_')[:-1])
            feature += '_lookback_' + feature_lookback
        if 'lookback' not in feature and 'future' in feature:
            feature_lookback = str(-int(feature.split('_')[-1]))
            feature = '_'.join(feature.split('_')[:-1])
            feature += '_lookahead_' + feature_lookback
    except:
        pass
        # print(feature)

    input_label = ['_lookahead_%d' % i for i in range(20)]
    output_label = ['lookahead : %d' % i for i in range(20)]
    lookahead_mapper = [(in_l, out_l) for in_l, out_l in zip(input_label, output_label)]

    input_label = ['_lookback_%d' % i for i in range(20)]
    output_label = ['lookback : %d' % i for i in range(20)]
    lookback_mapper = [(in_l, out_l) for in_l, out_l in zip(input_label, output_label)]

    label = []
    lookahead_mapper = [(in_l, out_l) for (in_l, out_l) in lookahead_mapper if in_l in feature]
    if len(lookahead_mapper) > 0:
        if len(lookahead_mapper) > 1:
            print(lookahead_mapper)
            lookahead_mapper = lookahead_mapper[-1:]
        assert len(lookahead_mapper) == 1, (feature, lookahead_mapper)
        label += [lookahead_mapper[0][1]]
        feature = feature.replace(lookahead_mapper[0][0], '')

    lookback_mapper = [(in_l, out_l) for (in_l, out_l) in lookback_mapper if in_l in feature]
    if len(lookback_mapper) > 0:
        if len(lookback_mapper) > 1:
            print(lookback_mapper)
            lookback_mapper = lookback_mapper[-1:]

        assert len(lookback_mapper) == 1, (feature, lookback_mapper)
        label += [lookback_mapper[0][1]]
        feature = feature.replace(lookback_mapper[0][0], '')

    switch_mapper = [(in_l, out_l) for (in_l, out_l) in switch_mapper if in_l in feature]
    if len(switch_mapper) > 0:
        if len(switch_mapper) > 1:
            print(switch_mapper)
            switch_mapper = switch_mapper[-1:]
        assert len(switch_mapper) == 1, (feature, switch_mapper)
        label += [switch_mapper[0][1]]
        feature = feature.replace(switch_mapper[0][0], '')

    if len(feature.split('_tput_predictor_')) > 1:
        throughput_predictor = feature.split('_tput_predictor_')[-1].replace('+_normalized', '')
        throughput_predictor = throughput_predictor.replace('percentile_q', 'Quantile')  # Percentile abbrevation
        throughput_predictor = throughput_predictor.replace("exponential_weighted_moving_average",
                                                            'EWMA')  ## Can be expanded if you have more features
        throughput_predictor = throughput_predictor.replace('_', ' ')
        feature = feature.split('_tput_predictor_')[0]
        label += ['Throughput Estimator : %s' % throughput_predictor]

    label = [value, 'Feature Type : %s' % feature_type_mapper[feature]] + label

    label += [class_label_mapper[class_label]]

    label = '<' + '<br/>'.join(label) + '>'  ## Join all labels
    return label
