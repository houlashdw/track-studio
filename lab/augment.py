# все промежуточные тензоры — в выходы модели (для поиска первого NaN)
import sys, onnx
from onnx import shape_inference
m = onnx.load(sys.argv[1])
try:
    mi = shape_inference.infer_shapes(m)
except Exception as e:
    print('shape inference failed', e); mi = m
types = {vi.name: vi.type for vi in list(mi.graph.value_info) + list(mi.graph.output) + list(mi.graph.input)}
outs = set(o.name for o in m.graph.output); order = []
for n in m.graph.node:
    for o in n.output:
        if o and o not in outs and o in types:
            vi = onnx.ValueInfoProto(); vi.name = o; vi.type.CopyFrom(types[o]); m.graph.output.append(vi); outs.add(o); order.append([o, n.op_type, n.name])
onnx.save(m, sys.argv[2])
import json; json.dump(order, open(sys.argv[3], 'w')); print('added', len(order))
