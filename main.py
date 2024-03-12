import argparse
import copy
import itertools
import json
import logging
import sys
import re
import threading
import time
import types

import numpy as np
import svgpathtools
import cadquery as cq

import gerbonara
import gerbonara.graphic_objects

from typing import List, Union, Tuple, Literal
from pathlib import Path
from warnings import warn
from itertools import groupby

logger = logging.getLogger(__name__)

#
# Benchmark
#

_benchmark_TLS = threading.local()


class benchmark(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.previous = getattr(_benchmark_TLS, benchmark.__class__.__name__, None)
        self.level = self.previous.level + 1 if self.previous is not None else 0
        tab = "\t" * self.level
        logger.debug(f"{tab}{self.name} : Starting...")
        self.start = time.time()
        setattr(_benchmark_TLS, benchmark.__class__.__name__, self)

    def __exit__(self, ty, val, tb):
        elapsed = time.time() - self.start
        tab = "\t" * self.level
        logger.debug(f"{tab}{self.name} : {elapsed:.3f} seconds")
        setattr(_benchmark_TLS, benchmark.__class__.__name__, self.previous)


#
# Math
#


def _euclid_to_tuple(p, dim3=False):
    if dim3:
        return np.real(p), np.imag(p), 0
    else:
        return np.real(p), np.imag(p)


def _tuple_to_euclid(p):
    return p[0] + p[1] * 1j


#
# SVG
#

def _divide_path(path, sub_div=100, endpoint=False):
    ps = []
    for j, s in enumerate(path):
        for i in np.linspace(0, 1, sub_div if not isinstance(s, svgpathtools.Line) else 2,
                             endpoint=(j == len(path) - 1 and endpoint)):
            p = s.point(i)
            ps.append(p)
    return ps


def _to_line_path(path, sub_div=100):
    if isinstance(path, Tuple):
        return _to_line_path(path[0], sub_div), path[1]
    if sub_div < 1:
        return path
    points = _divide_path(path, sub_div)
    ret = svgpathtools.Path(*[svgpathtools.Line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)])
    ret.closed = True
    return ret


def _get_angle(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    if np.isclose(v1, v2).all():
        return 0
    x = np.arccos(np.dot(v1, v2))
    return x if np.cross(v2, v1) < 0 else -x


def _is_path_clockwise(path):
    if not path.isclosed():
        raise Exception("Should be a closed path")
    path = [_euclid_to_tuple(p) for p in _divide_path(path, sub_div=20)]

    acc = 0
    for i in range(-1, len(path) - 2):
        v1 = np.subtract(path[i + 1], path[i + 0])
        v2 = np.subtract(path[i + 2], path[i + 1])
        s_sum = np.linalg.norm(v1) + np.linalg.norm(v2)
        angle = _get_angle(v1, v2)
        if angle > 0:
            acc += s_sum
        elif angle < 0:
            acc -= s_sum

    return acc < 0


def _close_path_sanitizing(path: Union[svgpathtools.Path, Tuple]):
    if isinstance(path, Tuple):
        return _close_path_sanitizing(path[0]), path[1]

    if path is None:
        return None

    # Create a copy
    path = copy.deepcopy(path)

    # Remove zero length segments
    path = svgpathtools.Path(*list(filter(lambda p: not np.isclose(np.linalg.norm(p.end - p.start), 0), path)))

    # Replace degenerate CubicBezier
    def replace_path(p):
        if isinstance(p, svgpathtools.CubicBezier):
            pts = np.asarray([_euclid_to_tuple(point) for point in [p.start, p.control1, p.control2, p.end]])
            if np.all(pts[:, 1] == pts[:, 1][0]) or np.all(pts[:, 0] == pts[:, 0][0]):
                warn(f'Replacing degenerate CubicBezier with a Line: CubicBezier(start={p.start},' +
                     f' control1={p.control1}, control2={p.control2}, end={p.end})' +
                     f' --> Line(start={p.start}, end={p.end})')
                return svgpathtools.Line(p.start, p.end)
        return p

    path = svgpathtools.Path(*[replace_path(p) for p in path])

    # Join segments
    for i in range(-1, len(path) - 1):
        found = False
        for j in range(i + 1, len(path)):
            if np.isclose(path[i].end, path[j].start):
                if j != i + 1:
                    path[i + 1], path[j] = path[j], path[i + 1]  # Swap
                found = True
                break
            if np.isclose(path[i].end, path[j].end):
                path[j] = path[j].reversed()
                if j != i + 1:
                    path[i + 1], path[j] = path[j], path[i + 1]  # Swap
                found = True
                break
        if not found:
            raise Exception("Can't close path")
        path[i].end = path[i + 1].start = (path[i].end + path[i + 1].start) / 2
    length = len(path)

    # Remove useless lines
    while True:
        i = -1
        while i < len(path) - 1:
            if path[i] == path[i + 1].reversed():
                del path[i + 1]
                del path[i]
            else:
                i += 1
        if len(path) == length:
            break
        length = len(path)

    path.closed = True
    return path.reversed() if _is_path_clockwise(path) else path


def _find_intersections(segment1, segment2, samples=50, ta=(0.0, 1.0, None), tb=(0.0, 1.0, None),
                        depth=0, enhancements=2, enhance_samples=50):
    """
    Calculate intersections by linearized polyline intersections with enhancements.
    We calculate probable intersections by linearizing our segment into `sample` polylines
    we then find those intersecting segments and the range of t where those intersections
    could have occurred and then subdivide those segments in a series of enhancements to
    find their intersections with increased precision.

    This code is fast, but it could fail by both finding a rare phantom intersection (if there
    is a low or no enhancements) or by failing to find a real intersection. Because the polylines
    approximation did not intersect in the base case.

    At a resolution of about 1e-15 the intersection calculations become unstable and intersection
    candidates can duplicate or become lost. We terminate at that point and give the last best
    guess.

    :param segment1:
    :param segment2:
    :param samples:
    :param ta:
    :param tb:
    :param depth:
    :param enhancements:
    :param enhance_samples:
    :return:
    """
    if depth == 0:
        # Quick Fail. There are no intersections without overlapping quick bounds
        try:
            s1x = [e.real for e in segment1.bpoints()]
            s2x = [e.real for e in segment2.bpoints()]
            if min(s1x) > max(s2x) or max(s1x) < min(s2x):
                return
            s1y = [e.imag for e in segment1.bpoints()]
            s2y = [e.imag for e in segment2.bpoints()]
            if min(s1y) > max(s2y) or max(s1y) < min(s2y):
                return
        except AttributeError:
            pass
    assert (samples >= 2)
    a = np.linspace(ta[0], ta[1], num=samples)
    b = np.linspace(tb[0], tb[1], num=samples)
    step_a = a[1] - a[0]
    step_b = b[1] - b[0]
    j = segment1.points(a) if hasattr(segment1, 'points') else [segment1.point(i) for i in a]
    k = segment2.points(b) if hasattr(segment2, 'points') else [segment2.point(i) for i in b]

    ax1, bx1 = np.meshgrid(np.real(j[:-1]), np.real(k[:-1]))
    ax2, bx2 = np.meshgrid(np.real(j[1:]), np.real(k[1:]))
    ay1, by1 = np.meshgrid(np.imag(j[:-1]), np.imag(k[:-1]))
    ay2, by2 = np.meshgrid(np.imag(j[1:]), np.imag(k[1:]))

    denom = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    qa = (bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)
    qb = (ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)
    hits = np.dstack(
        (
            denom != 0,  # Cannot be parallel.
            np.sign(denom) == np.sign(qa),  # D and Qa must have same sign.
            np.sign(denom) == np.sign(qb),  # D and Qb must have same sign.
            abs(denom) >= abs(qa),  # D >= Qa (else not between 0 - 1)
            abs(denom) >= abs(qb),  # D >= Qb (else not between 0 - 1)
        )
    ).all(axis=2)

    where_hit = np.argwhere(hits)
    if len(where_hit) != 1 and step_a < 1e-10:
        # We're hits are becoming unstable give last best value.
        if ta[2] is not None and tb[2] is not None:
            yield ta[2], tb[2]
        return

    # Calculate the t values for the intersections
    ta_hit = qa[hits] / denom[hits]
    tb_hit = qb[hits] / denom[hits]

    for i, hit in enumerate(where_hit):

        at = ta[0] + float(hit[1]) * step_a  # Zoomed min+segment intersected.
        bt = tb[0] + float(hit[0]) * step_b
        a_fractional = ta_hit[i] * step_a  # Fractional guess within intersected segment
        b_fractional = tb_hit[i] * step_b
        if depth == enhancements:
            # We've enhanced as good as we can, yield the current + segment t-value to our answer
            yield at + a_fractional, bt + b_fractional
        else:
            yield from _find_intersections(
                segment1,
                segment2,
                ta=(at, at + step_a, at + a_fractional),
                tb=(bt, bt + step_b, bt + b_fractional),
                samples=enhance_samples,
                depth=depth + 1,
                enhancements=enhancements,
                enhance_samples=enhance_samples,
            )


def _offset_curve(path, offset_distance):
    """Takes in a Path object, `path`, and a distance,
    `offset_distance`, and outputs a piecewise-linear approximation
    of the 'parallel' offset curve."""
    sign = 1 if offset_distance >= 0 else -1

    # No inner?
    if offset_distance < 0:
        xmin, xmax, ymin, ymax = path.bbox()
        if -offset_distance * 2 >= abs(xmin - xmax) or -offset_distance * 2 >= abs(ymin - ymax):
            return None

    def new_vector(va1, va2):
        va3 = va1 + va2
        va3 /= np.linalg.norm(va3)
        d = offset_distance / np.sqrt((1 + np.dot([va1.real, va1.imag], [va2.real, va2.imag])) / 2)
        va3 *= d
        return va3

    new_path = []
    for i in range(len(path)):
        seg1 = path[i - 1]
        seg2 = path[i]
        seg3 = path[(i + 1) % len(path)]

        v1 = new_vector(seg1.normal(1), seg2.normal(0))
        v2 = new_vector(seg2.normal(1), seg3.normal(0))

        start = seg2.start + v1
        end = seg2.end + v2
        if isinstance(seg2, svgpathtools.Line):
            new_path.append(svgpathtools.Line(start, end))
        elif isinstance(seg2, svgpathtools.Arc):
            r = offset_distance
            new_path.append(
                svgpathtools.Arc(start, seg2.radius + _tuple_to_euclid((r, r)), seg2.rotation, seg2.large_arc,
                                 seg2.sweep, end))
        elif isinstance(seg2, svgpathtools.CubicBezier):
            a = v1 + (v2 / np.linalg.norm(v2) * (offset_distance * 0.5522847498 * sign)) if offset_distance != 0 else 0
            b = v2 + (v1 / np.linalg.norm(v1) * (offset_distance * 0.5522847498 * sign)) if offset_distance != 0 else 0
            new_path.append(svgpathtools.CubicBezier(start, seg2.control1 + a, seg2.control2 + b, end))
        else:
            raise Exception("Not supported: {0}".format(type(seg2)))

    i = 0
    while i < len(new_path):
        j = i + 2
        while j < i + len(new_path) / 2 + 1:
            k = j % len(new_path)
            r = list(_find_intersections(new_path[i], new_path[k], samples=50))
            if len(r) == 1:
                new_path[i] = new_path[i].cropped(0, r[0][0])
                new_path[k] = new_path[k].cropped(r[0][1], 1)
                a = max(j - len(new_path), 0)
                new_path = new_path[a:i + 1] + new_path[j:]
                i = i - a
                j = i + 2
            else:
                j += 1
        i += 1

    offset_path = svgpathtools.Path(*new_path)
    return offset_path


def _to_svg(seg, arc_poly=False):
    if isinstance(seg, list):
        segments = list(itertools.chain(
            *[path for path, attributes in [_to_svg(x, arc_poly=arc_poly) for x in seg] if path is not None]))
        return svgpathtools.Path(*segments), {}
    if isinstance(seg, gerbonara.graphic_objects.Line) and seg.p1 == seg.p2:
        return None, None
    primitives = seg.as_primitives()
    assert len(primitives) == 1
    obj = primitives[0]
    if arc_poly:
        obj = obj.to_arc_poly()
    svg = obj.to_svg()
    paths, attributes = svgpathtools.svgstr2paths(str(svg))
    assert len(paths) == len(attributes) == 1
    return paths[0], attributes[0]


#
# CadQuery
#

def _to_wire(path, support: Union[cq.Workplane, cq.Sketch] = cq.Workplane("front"),
             mode: Literal["a", "s", "i", "c"] = "a", wire_transform=None):
    def convert(seg):
        if isinstance(seg, svgpathtools.Line):
            return cq.Edge.makeLine(_euclid_to_tuple(seg.start), _euclid_to_tuple(seg.end))
        elif isinstance(seg, svgpathtools.Arc):
            return cq.Edge.makeThreePointArc(
                _euclid_to_tuple(seg.start),
                _euclid_to_tuple(seg.point(0.5)),
                _euclid_to_tuple(seg.end)
            )
        elif isinstance(seg, svgpathtools.CubicBezier):
            return cq.Edge.makeSpline([
                cq.Vector(*_euclid_to_tuple(seg.start, True)),
                cq.Vector(*_euclid_to_tuple(seg.point(1 / 3), True)),
                cq.Vector(*_euclid_to_tuple(seg.point(2 / 3), True)),
                cq.Vector(*_euclid_to_tuple(seg.end, True))
            ])
        else:
            raise Exception("Not supported: {0}".format(type(seg)))

    edges = [convert(seg) for seg in path]
    wires = cq.Wire.combine(edges)
    assert len(wires) == 1
    wire = wires[0]
    if wire_transform is not None:
        wire = wire_transform(wire)

    if isinstance(support, cq.Workplane):
        sketch = cq.Sketch().face(wire)
        ret = support.placeSketch(sketch)
    elif isinstance(support, cq.Sketch):
        ret = support.face(wire, mode=mode)
    else:
        raise RuntimeError(f"Unsupported support type: {type(support)}")
    return ret


def _normalize_faces(sketch):
    sketch._faces = cq.Compound.makeCompound(
        [cq.Face(f.wrapped.Reversed()) if f.normalAt().z < 0 else f for f in sketch.faces()])
    return sketch


def _paths_union(paths, wire_transform=None):
    """
    Divide and conquer union. The first paths can be huge, making the successive union really slow.
    With this algorithm that increase the chances to fast simple union.
    :param paths: paths to union
    :return: one sketch
    """
    if len(paths) < 2:
        return _to_wire(paths[0], cq.Sketch(), "a", wire_transform=wire_transform)

    a = _paths_union(paths[:len(paths) // 2], wire_transform=wire_transform)
    b = _paths_union(paths[len(paths) // 2:], wire_transform=wire_transform)
    c = a.face(b, mode="a")
    return c


#
#
#

def _get_args(gbr_files: List[Path], configuration):
    def get_regex(v):
        x = re.escape(v)
        return x.replace(re.escape("{board_name}"), '(?P<board_name>.*)')

    args = dict()
    args['overrides'] = overrides = {get_regex(value): key for key, value in configuration.mapping.__dict__.items()}

    for fr in overrides.keys():
        for f in gbr_files:
            res = re.fullmatch(fr, f.name)
            if res:
                args['board_name'] = res.group("board_name")
                break

    return args


def _get_files(args):
    gbr_files = []
    for path_object in Path(args.source).rglob("*"):
        if path_object.is_file():
            gbr_files.append(path_object)
    return gbr_files


def _create_board(configuration, svg, npth, pth):
    body_o = cq.Workplane("front")

    with benchmark("Create outline"):
        sketch = _to_wire(svg[0], cq.Sketch())

    with benchmark("Union holes"):
        union_holes = _paths_union([path for path, _ in npth + pth])
        sketch = sketch.face(union_holes, mode="s")

    with benchmark("Clean"):
        sketch = _normalize_faces(sketch)  # Needed for correct clean
        sketch = sketch.clean()

    with benchmark("Extrude board"):
        body_o = body_o.placeSketch(sketch).extrude(configuration.pcb.thickness, True)
    return body_o


def _create_copper(configuration, svgs, offset, npth, pth):
    body_o = cq.Workplane("front", origin=(0, 0, offset))

    def get_combine(path, attributes):
        if path is None or 'fill' not in attributes:
            return None
        return "a" if attributes['fill'] == 'black' else "s"

    with benchmark("Clean SVG"):
        path_with_combine = [(x, get_combine(x, y)) for x, y in svgs]
        path_with_combine = list(filter(lambda x: x[1] is not None, path_with_combine))

    with benchmark("Split by combine mode"):
        groups = [(mode, list(zip(*g))[0]) for mode, g in groupby(path_with_combine, lambda x: x[1])]

    with benchmark("Union groups"):
        groups = [(mode, _paths_union(g)) for mode, g in groups]

    with benchmark("Final union"):
        sketch = groups.pop(0)[1]
        for mode, group in groups:
            sketch = sketch.face(group, mode=mode)

    with benchmark("Union holes"):
        holes = npth + pth
        union_holes = _paths_union([path for path, _ in holes])
        sketch = sketch.face(union_holes, mode="s")

    with benchmark("Clean"):
        sketch = _normalize_faces(sketch)  # Needed for correct clean
        sketch = sketch.clean()

    with benchmark("Extrude copper"):
        body_o = body_o.placeSketch(sketch).extrude(configuration.copper.thickness, combine=False)
    return body_o


def _create_via(configuration, pth):
    body_o = cq.Workplane("front", origin=(0, 0, -configuration.copper.thickness))

    with benchmark("Via outside walls"):
        outside = _paths_union([path for path, _ in pth])

    with benchmark("Via inside walls"):
        inside = _paths_union([_offset_curve(path, -configuration.via.plating.thickness) for path, _ in pth])

    with benchmark("Create holes"):
        sketch = outside.face(inside, mode="s")

    with benchmark("Clean"):
        sketch = _normalize_faces(sketch)  # Needed for correct clean
        sketch = sketch.clean()

    with benchmark("Extrude copper"):
        body_o = body_o.placeSketch(sketch).extrude(configuration.pcb.thickness + configuration.copper.thickness * 2,
                                                    combine=False)

    return body_o


def _get_color(color):
    return cq.Color(*[c / 255 for c in color])


def app(args):
    with open(args.configuration) as configuration_file:
        configuration = json.load(configuration_file, object_hook=lambda dct: types.SimpleNamespace(**dct))

    gbr_files = _get_files(args)

    copper_color = _get_color(configuration.copper.color)
    pcb_color = _get_color(configuration.pcb.color)

    with benchmark("Load gerber files"):
        stack = gerbonara.LayerStack.from_files(gbr_files, autoguess=False, **_get_args(gbr_files, configuration))

    svgs = {}
    with benchmark("To SVG"):
        svgs['outline'] = _close_path_sanitizing(_to_svg(stack.outline.objects, arc_poly=False))
        svgs['top_copper'] = [_close_path_sanitizing(_to_svg(o, arc_poly=True)) for o in
                              stack.top_side['top copper'].objects]
        svgs['bottom copper'] = [_close_path_sanitizing(_to_svg(o, arc_poly=True)) for o in
                                 stack.bottom_side['bottom copper'].objects]
        svgs['drill_npth'] = [_close_path_sanitizing(_to_svg(o, arc_poly=True)) for o in stack.drill_npth.objects]
        svgs['drill_pth'] = [_to_line_path(_close_path_sanitizing(_to_svg(o, arc_poly=True)), configuration.via.subdiv)
                             for o in stack.drill_pth.objects]

    elements = {}
    attributes = {}

    with benchmark("Create board"):
        elements['board'] = _create_board(configuration, svgs['outline'], svgs['drill_npth'], svgs['drill_pth'])
        attributes['board'] = dict(color=pcb_color)

    with benchmark("Top copper"):
        elements['top_copper'] = _create_copper(configuration, svgs['top_copper'], configuration.pcb.thickness,
                                                svgs['drill_npth'], svgs['drill_pth'])
        attributes['top_copper'] = dict(color=copper_color)

    with benchmark("Bottom copper"):
        elements['bottom_copper'] = _create_copper(configuration, svgs['bottom copper'],
                                                   -configuration.copper.thickness, svgs['drill_npth'],
                                                   svgs['drill_pth'])
        attributes['bottom_copper'] = dict(color=copper_color)

    with benchmark("Vias"):
        elements['vias'] = _create_via(configuration, svgs['drill_pth'])
        attributes['vias'] = dict(color=copper_color)

    assembly = cq.Assembly(name="PCB")
    for name, o in elements.items():
        assembly.add(o, name=name, **attributes.get(name, {}))

    with benchmark("Export"):
        assembly.save(args.output)

    try:
        show_object(assembly, name='PCB')
    except BaseException:
        pass


def _main(argv=sys.argv):
    def auto_int(x):
        return int(x, 0)

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                        format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(prog=argv[0], description='Gerber2Step',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose_count', action='count', default=0,
                        help="increases log verbosity for each occurrence.")
    parser.add_argument('-c', '--configuration', default="config.json",
                        help="configuration file")
    parser.add_argument('source',
                        help="source")
    parser.add_argument('output',
                        help="STEP output file")

    # Parse
    args, unknown_args = parser.parse_known_args(argv[1:])

    # Set logging level
    logging.getLogger().setLevel(max(3 - args.verbose_count, 0) * 10)

    app(args)

    return 0


# ------------------------------------------------------------------------------

def main():
    try:
        sys.exit(_main(sys.argv))
    except Exception as e:
        logger.exception(e)
        sys.exit(-1)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
