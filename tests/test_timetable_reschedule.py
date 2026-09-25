"""班次重排核心算法测试。

确定性用例覆盖：零平移可行、逐班贪心选最近时刻失败（误判无解或成本更高，
必须动态规划）、同成本字典序最早、headway 无解与恰好满足、半开窗口边界
（窗口右端与发车范围上界均不含）、多路口绿波带联动、行驶时间波动收缩、
tolerance=0、50 个班次上限及 3600 秒大范围性能。另含数千组随机参数下与
独立暴力动态规划（O(n·W²) 逐点转移）的逐项对照。
"""

import random
import time

from app.greenwave import compute_timetable_reschedule, compute_windows


def it(**kw):
    # 默认 green_start == green_end 表示整周期常绿，所有取值均满足 schema 约束。
    return {
        "cycle": 60, "offset": 0, "green_start": 0, "green_end": 0,
        "travel": 0, **kw,
    }


# ---------- 确定性用例 ----------

def test_reschedule_unchanged_when_already_feasible():
    # 绿灯 [0,30) 周期 60，两班原时刻 5、15 本身都在窗口内且间隔 10：零平移
    nodes = [it(green_start=0, green_end=30)]
    assert compute_timetable_reschedule(
        nodes, 0, 120, [5, 15], tolerance=30, headway=10
    ) == [5, 15]


def test_reschedule_greedy_nearest_falsely_infeasible():
    # 绿灯 [0,20) 周期 60：窗口 [0,20)、[60,80)。
    # p=[12,19]、tol=10、h=10。
    # 逐班贪心：首班取最近的 12（零平移）⇒ 次班须 >=22，而次班允许域
    # [9,29] 内没有任何窗口（下一窗口 60 超出 29）⇒ 贪心误判无解。
    # 动态规划：首班取 9（平移 3）⇒ 次班取 19（平移 0），间隔 10，可行，
    # 总成本 3，为最优解。
    nodes = [it(green_start=0, green_end=20)]
    assert compute_timetable_reschedule(
        nodes, 0, 120, [12, 19], tolerance=10, headway=10
    ) == [9, 19]


def test_reschedule_greedy_nearest_feasible_but_costlier():
    # 绿灯 [0,10) 周期 40：窗口 [0,10)、[40,50)、[80,90)。
    # p=[25,50]、tol=40、h=10。
    # 逐班贪心：首班最近可行点为 40（平移 15）⇒ 次班须 >=50，跳过 [40,50)
    # 整窗只能取 80（平移 30），总成本 45，且贪心看不到更优配对。
    # 动态规划：首班取 9（平移 16）⇒ 次班取 49（平移 1），间隔 40>=10，
    # 总成本 17，远优于贪心的 45。
    nodes = [it(cycle=40, green_start=0, green_end=10)]
    assert compute_timetable_reschedule(
        nodes, 0, 100, [25, 50], tolerance=40, headway=10
    ) == [9, 49]


def test_reschedule_look_ahead_to_previous_window():
    # 同上窗口结构；p=[25,46]、tol=20、h=10。
    # 逐班贪心：首班最近点 40（平移 15）⇒ 次班须 >=50，允许域 [26,66] 内
    # [40,50) 已被越过、下一窗 80 超出 66 ⇒ 贪心误判无解。
    # 动态规划：首班回退到 9（平移 16）⇒ 次班取 46（平移 0），总成本 16。
    nodes = [it(cycle=40, green_start=0, green_end=10)]
    assert compute_timetable_reschedule(
        nodes, 0, 100, [25, 46], tolerance=20, headway=10
    ) == [9, 46]


def test_reschedule_lexicographically_earliest_on_tie_single_trip():
    # 绿灯 [0,9) 周期 12：窗口 [0,9)、[12,21)。单班 p=10：
    # tol=2 时允许域 [8,12] 内可行点为 8（距 2）与 12（距 2），成本相同，
    # 字典序最早取 8。
    nodes = [it(cycle=12, green_start=0, green_end=9)]
    assert compute_timetable_reschedule(
        nodes, 0, 24, [10], tolerance=2, headway=1
    ) == [8]
    # tol=1：允许域 [9,11] 不含任何窗口（9 是半开端点、下一窗从 12 起）⇒ 无解；
    # tol=3：域 [7,13] 内 8 距 2、12 距 2 仍打平取 8；p=9、tol=1 时可行点
    # 为 8（距 1）与 12（距 3），取 8。
    assert compute_timetable_reschedule(
        nodes, 0, 24, [10], tolerance=1, headway=1
    ) is None
    assert compute_timetable_reschedule(
        nodes, 0, 24, [10], tolerance=3, headway=1
    ) == [8]
    assert compute_timetable_reschedule(
        nodes, 0, 24, [9], tolerance=1, headway=1
    ) == [8]


def test_reschedule_lexicographically_earliest_on_tie_two_trips():
    # 常绿路口，范围 [0,20)，p=[4,6]、tol=8、h=10。
    # 各可行配对（成本相同的最优族）：d0∈[0,3] 时 d1=10，成本
    # (4-d0)+4 = 8-d0；d0=0 时成本 8，d0=3 时成本 5——成本并不相等。
    # 逐一枚举：d0=0,d1=10 成本 4+4=8；d0=1,d1=11 成本 3+5=8；
    # d0=2,d1=12 成本 2+6=8；d0=3,d1=13 成本 1+7=8；d0=4,d1=14
    # 成本 0+8=8。全部成本均为 8，字典序最早为 [0,10]。
    nodes = [it()]
    assert compute_timetable_reschedule(
        nodes, 0, 20, [4, 6], tolerance=8, headway=10
    ) == [0, 10]


def test_reschedule_headway_violation_returns_none():
    # 常绿路口：p=[0,5]、tol=0、h=10：只能取原时刻，间隔 5 < 10 ⇒ 无解
    nodes = [it()]
    assert compute_timetable_reschedule(nodes, 0, 60, [0, 5], 0, 10) is None


def test_reschedule_headway_met_at_exact_gap():
    # 常绿：p=[0,5]、tol=5、h=10；[0,10] 成本 5，[5,15] 成本 10，
    # 最优且字典序最早为 [0,10]（间隔恰为 10，允许）
    nodes = [it()]
    assert compute_timetable_reschedule(nodes, 0, 60, [0, 5], 5, 10) == [0, 10]


def test_reschedule_half_open_window_end_excluded():
    # 绿灯 [0,20)：t=20 遇红。单班 p=20、tol=0 无解；tol=1 时最近可行点 19
    nodes = [it(green_start=0, green_end=20)]
    assert compute_timetable_reschedule(nodes, 0, 60, [20], 0, 1) is None
    assert compute_timetable_reschedule(nodes, 0, 60, [20], 1, 1) == [19]
    # 发车范围上界 depart_to 同样不含：范围 [0,20) 内 p=19 仍可取，
    # 范围 [0,19) 时 19 越界只能取 18
    assert compute_timetable_reschedule(nodes, 0, 20, [19], 1, 1) == [19]
    assert compute_timetable_reschedule(nodes, 0, 19, [18], 1, 1) == [18]


def test_reschedule_two_intersections_green_band():
    # README 示例的两路口绿波带：公共窗口 [60q, 60q+30)
    nodes = [
        it(green_start=0, green_end=30, travel=0, offset=0),
        it(green_start=0, green_end=30, travel=30, offset=30),
    ]
    # 两班原时刻 35、95 均在红段 [30,60)，tol=10、h=10：
    # 班0 允许域 [25,45]∩窗=[25,30)，班1 允许域 [85,105]∩窗=[85,90)；
    # 各班取最靠近原时刻的点 29 与 89，总成本 12（间隔 60>=10）。
    assert compute_timetable_reschedule(
        nodes, 0, 300, [35, 95], tolerance=10, headway=10
    ) == [29, 89]


def test_reschedule_travel_max_shrinks_feasible_points():
    # 绿灯 [20,50)、行驶 [0,10]：公共窗口右端收缩为 [20,40) 周期 60。
    # 单班 p=45、tol=10：允许 [35,55]∩窗=[35,40)，最近点 39（成本 6）；
    # 40 是收缩后窗口右端（不含），不可取。
    nodes = [it(green_start=20, green_end=50, travel=0, travel_max=10)]
    assert compute_timetable_reschedule(nodes, 0, 120, [45], 10, 1) == [39]


def test_reschedule_no_common_window_returns_none():
    # 两路口窗口互不相交：任何班次都无可行点
    nodes = [
        it(green_start=0, green_end=10),
        it(green_start=20, green_end=30),
    ]
    assert compute_timetable_reschedule(
        nodes, 0, 60, [5, 25], 60, 1
    ) is None


def test_reschedule_tolerance_zero_requires_exact_green():
    # tol=0：原时刻不在窗口内即无解；在窗口内且间隔足够则原样返回
    nodes = [it(green_start=0, green_end=30)]
    assert compute_timetable_reschedule(nodes, 0, 120, [5, 15], 0, 10) == [
        5, 15
    ]
    assert compute_timetable_reschedule(nodes, 0, 120, [35], 0, 1) is None


def test_reschedule_fifty_trips_unchanged():
    # 50 个班次（上限）：常绿路口，原时刻每隔 3 秒，h=3、tol=0 ⇒ 原样
    nodes = [it()]
    originals = [3 * k for k in range(50)]  # 0,3,...,147
    assert compute_timetable_reschedule(
        nodes, 0, 300, originals, tolerance=0, headway=3
    ) == originals


def test_reschedule_fifty_trips_uniform_shift():
    # 常绿，原时刻等距 2 秒（间隔不足 h=3），tol=60：最小总平移且字典序
    # 最早的时刻表为 0,3,6,...,147（各班相对原时刻的右移量 0..49 均在
    # tol 内；末班 147 < depart_to=600）
    nodes = [it()]
    originals = [2 * k for k in range(50)]
    assert compute_timetable_reschedule(
        nodes, 0, 600, originals, tolerance=60, headway=3
    ) == [3 * k for k in range(50)]


def test_reschedule_performance_full_range_fifty_trips():
    # 极端规模：3600 秒范围、50 个班次、tol 拉满，须在秒级完成
    nodes = [it()]
    originals = [60 + 70 * k for k in range(50)]
    started = time.perf_counter()
    result = compute_timetable_reschedule(
        nodes, 0, 3600, originals, tolerance=3600, headway=71
    )
    elapsed = time.perf_counter() - started
    assert result is not None
    assert all(result[k] - result[k - 1] >= 71 for k in range(1, 50))
    assert elapsed < 5.0


# ---------- 随机参数与独立暴力 DP 对照 ----------


def _brute_force_reschedule(nodes, lo, hi, originals, tol, h):
    """独立参考实现：窗口内逐点 O(n·W²) 动态规划，不做后缀最小值优化。

    状态只存最小成本（与被测实现相同的目标函数），再自前向后在可行点中
    按 (成本, 出发时刻) 取最小者恢复——即同成本下的字典序最早序列。
    """
    windows = compute_windows(nodes, lo, hi)
    if not windows:
        return None

    point_sets: list[list[int]] = []
    for p in originals:
        points: list[int] = []
        for start, end in windows:
            a = max(start, p - tol, lo)
            b = min(end, p + tol + 1, hi)
            if a < b:
                points.extend(range(a, b))
        if not points:
            return None
        point_sets.append(points)

    INF = 10**18
    costs: dict[int, int] = {
        t: abs(t - originals[-1]) for t in point_sets[-1]
    }
    layers = [costs]
    for k in range(len(originals) - 2, -1, -1):
        current: dict[int, int] = {}
        successor = layers[-1]
        succ_points = sorted(successor)
        for t in point_sets[k]:
            best = INF
            threshold = t + h
            for u in succ_points:
                if u < threshold:
                    continue
                value = abs(t - originals[k]) + successor[u]
                if value < best:
                    best = value
            if best < INF:
                current[t] = best
        if not current:
            return None
        layers.append(current)
    layers.reverse()

    total = min(layers[0].values())
    result: list[int] = []
    lower = lo
    for k, layer in enumerate(layers):
        chosen = min(
            t for t in layer
            if t >= lower and layer[t] == total
        )
        result.append(chosen)
        lower = chosen + h
        total -= abs(chosen - originals[k])
    return result


def test_matches_brute_force_random():
    random.seed(20260925)
    checked = 0
    for _ in range(3000):
        n_nodes = random.randint(1, 5)
        nodes = []
        for _ in range(n_nodes):
            c = random.randint(1, 40)
            travel = random.randint(0, 40)
            kwargs = dict(
                cycle=c,
                offset=random.randint(0, c - 1),
                green_start=random.randint(0, c - 1),
                green_end=random.randint(0, c - 1),
                travel=travel,
            )
            if random.random() < 0.5:
                kwargs["travel_max"] = travel + random.randint(0, 12)
            nodes.append(it(**kwargs))

        lo = random.randint(0, 100)
        hi = lo + random.randint(10, 160)

        n = random.randint(1, 6)
        if hi - lo < n:
            continue
        originals = sorted(random.sample(range(lo, hi), n))
        tol = random.randint(0, 30)
        h = random.randint(1, 25)

        expected = _brute_force_reschedule(
            nodes, lo, hi, originals, tol, h
        )
        actual = compute_timetable_reschedule(
            nodes, lo, hi, originals, tol, h
        )
        assert actual == expected, (
            f"nodes={nodes} lo={lo} hi={hi} originals={originals} "
            f"tol={tol} h={h}: expected={expected} actual={actual}"
        )
        checked += 1
    assert checked > 1000
