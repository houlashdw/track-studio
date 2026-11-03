"""веса ONNX хранятся в float16 (файл вдвое меньше, <100 МБ для GitHub), считается всё в float32: перед использованием — Cast"""
import onnx, sys, numpy as np
from onnx import numpy_helper, helper, TensorProto
def half_store(src, dst, min_size=1024):
    m = onnx.load(src); g = m.graph; new_init = []; casts = []
    for t in list(g.initializer):
        if t.data_type == TensorProto.FLOAT and int(np.prod(t.dims)) >= min_size:
            a = numpy_helper.to_array(t).astype(np.float16)
            h = numpy_helper.from_array(a, t.name + '__h'); new_init.append(h)
            casts.append(helper.make_node('Cast', [t.name + '__h'], [t.name], to=TensorProto.FLOAT, name=t.name + '__cast'))
            g.initializer.remove(t)
    g.initializer.extend(new_init)
    for i, n in enumerate(casts): g.node.insert(i, n)
    onnx.save(m, dst)
if __name__ == '__main__': half_store(sys.argv[1], sys.argv[2])
