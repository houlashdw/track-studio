"""Piper (VITS) → модель с управляемыми длительностями: вход durations [1,1,N] (кадры по 256 отсчётов @22050; <0 — своя длительность модели),
выход dur_pred — длительности, которые модель предсказала сама. Так русскую строку можно положить точно в бит."""
import onnx, sys
from onnx import helper, TensorProto
def force(src, dst):
    m = onnx.load(src); g = m.graph
    ceil = [n for n in g.node if n.op_type == 'Ceil']; assert len(ceil) == 1, 'Ceil не один'
    c = ceil[0].output[0]
    g.input.append(helper.make_tensor_value_info('durations', TensorProto.FLOAT, [1, 1, 'phonemes']))
    zero = helper.make_tensor('dz_zero', TensorProto.FLOAT, [], [0.0]); g.initializer.append(zero)
    used = 'dur_used'
    nodes = [helper.make_node('Less', ['durations', 'dz_zero'], ['dur_isneg'], name='dur_less'),
             helper.make_node('Where', ['dur_isneg', c, 'durations'], [used], name='dur_where')]
    for n in g.node:
        if n is ceil[0]: continue
        for i, x in enumerate(n.input):
            if x == c: n.input[i] = used
    idx = list(g.node).index(ceil[0])
    for k, n in enumerate(nodes): g.node.insert(idx + 1 + k, n)
    g.output.append(helper.make_tensor_value_info(c, TensorProto.FLOAT, [1, 1, 'phonemes']))
    g.output[-1].name = c
    onnx.checker.check_model(m); onnx.save(m, dst)
if __name__ == '__main__': force(sys.argv[1], sys.argv[2])
