import graphviz
from graphviz import Digraph


def create_model_graph():
    dot = Digraph('STGCNMultiTask_Model', format='png')
    dot.attr(rankdir='LR', fontsize='12', fontname='Helvetica')

    # 设置全局节点属性
    dot.attr('node', shape='rectangle', style='filled', color='lightgrey', fontsize='10', fontname='Helvetica')

    # 定义主模块 STGCNMultiTask
    with dot.subgraph(name='cluster_STGCNMultiTask') as c:
        c.attr(style='filled', color='lightblue', label='STGCNMultiTask')

        # 定义子模块 STConv Layers
        with c.subgraph(name='cluster_STConvLayers') as sc:
            sc.attr(style='filled', color='lightyellow', label='STConv Layers')
            # 假设有多个 STConv 层，这里以两层为例
            sc.node('STConv1', 'STConv Layer 1')
            sc.node('STConv2', 'STConv Layer 2')
            sc.node('STConv3', 'STConv Layer 3')

        # 其他子模块
        c.node('STAttention', 'STAttention')
        c.node('Classifier', 'Classifier')
        c.node('KeyActionDetector', 'Key Action Detector')
        c.node('AngleFC', 'Angle FC')
        c.node('GlobalPool', 'Global Avg Pool')

    # 定义 STConv1 结构
    with dot.subgraph(name='cluster_STConv1') as sc1:
        sc1.attr(style='filled', color='lightgreen', label='STConv1')
        sc1.node('TemporalConv1_1', 'TemporalConv1')
        sc1.node('ChebConv1', 'ChebConv')
        sc1.node('TemporalConv2_1', 'TemporalConv2')

    # 定义 STConv2 结构
    with dot.subgraph(name='cluster_STConv2') as sc2:
        sc2.attr(style='filled', color='lightgreen', label='STConv2')
        sc2.node('TemporalConv1_2', 'TemporalConv1')
        sc2.node('ChebConv2', 'ChebConv')
        sc2.node('TemporalConv2_2', 'TemporalConv2')

    # 定义 STConv3 结构
    with dot.subgraph(name='cluster_STConv3') as sc3:
        sc3.attr(style='filled', color='lightgreen', label='STConv3')
        sc3.node('TemporalConv1_3', 'TemporalConv1')
        sc3.node('ChebConv3', 'ChebConv')
        sc3.node('TemporalConv2_3', 'TemporalConv2')

    # 连接主模块与子模块
    dot.edge('cluster_STConvLayers', 'cluster_STConv1')
    dot.edge('cluster_STConvLayers', 'cluster_STConv2')
    dot.edge('cluster_STConvLayers', 'cluster_STConv3')
    dot.edge('cluster_STConvLayers', 'STAttention')
    dot.edge('cluster_STConvLayers', 'Classifier')
    dot.edge('cluster_STConvLayers', 'KeyActionDetector')
    dot.edge('cluster_STConvLayers', 'AngleFC')
    dot.edge('cluster_STConvLayers', 'GlobalPool')

    # 连接 STConv1 子模块
    dot.edge('STConv1', 'TemporalConv1_1')
    dot.edge('STConv1', 'ChebConv1')
    dot.edge('STConv1', 'TemporalConv2_1')

    # 连接 STConv2 子模块
    dot.edge('STConv2', 'TemporalConv1_2')
    dot.edge('STConv2', 'ChebConv2')
    dot.edge('STConv2', 'TemporalConv2_2')

    # 连接 STConv3 子模块
    dot.edge('STConv3', 'TemporalConv1_3')
    dot.edge('STConv3', 'ChebConv3')
    dot.edge('STConv3', 'TemporalConv2_3')

    # 连接其他模块
    dot.edge('STAttention', 'Classifier')
    dot.edge('Classifier', 'KeyActionDetector')
    dot.edge('KeyActionDetector', 'AngleFC')
    dot.edge('AngleFC', 'GlobalPool')

    return dot


if __name__ == "__main__":
    model_graph = create_model_graph()
    model_graph.render('stgcn_multitask_structure', view=True)
    print("模型结构图已生成并保存在 'stgcn_multitask_structure.png'")
