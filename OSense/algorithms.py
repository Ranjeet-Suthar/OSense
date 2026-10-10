"""
OSense - algorithms.py
All the OS algorithm logic + the ML recommender lives here.
(app.py only takes care of the user interface.)

Topics covered
1. CPU Scheduling    : FCFS, SJF, Round Robin
2. Page Replacement  : FIFO, LRU, LFU
3. Disk Scheduling   : FCFS, SSTF, SCAN, C-SCAN   (new)
"""
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 1. CPU SCHEDULING
# each function returns (gantt, table)
#   gantt : list of (pid, start, end)
#   table : dict pid -> dict(AT, BT, CT, TAT, WT, RT)
# ---------------------------------------------------------------


def _build_table(arrival, burst, completion, gantt):
    # Response time = (first time the process gets the CPU) - arrival time
    first_start = {}
    for pid, s, e in gantt:
        if pid not in first_start:
            first_start[pid] = s
    table = {}
    for i in range(len(arrival)):
        tat = completion[i] - arrival[i]
        wt = tat - burst[i]
        rt = first_start[i + 1] - arrival[i]
        table[i + 1] = {"AT": arrival[i], "BT": burst[i],
                        "CT": completion[i], "TAT": tat, "WT": wt, "RT": rt}
    return table


def fcfs(arrival, burst):
    n = len(arrival)
    order = sorted(range(n), key=lambda i: arrival[i])
    time, gantt, comp = 0, [], [0] * n
    for i in order:
        if time < arrival[i]:
            time = arrival[i]            # CPU idle till process arrives
        gantt.append((i + 1, time, time + burst[i]))
        time += burst[i]
        comp[i] = time
    return gantt, _build_table(arrival, burst, comp, gantt)


def sjf(arrival, burst):
    """Non-preemptive Shortest Job First."""
    n = len(arrival)
    done, time, gantt, comp = [False] * n, 0, [], [0] * n
    while not all(done):
        ready = [i for i in range(n) if not done[i] and arrival[i] <= time]
        if not ready:
            time = min(arrival[i] for i in range(n) if not done[i])
            continue
        i = min(ready, key=lambda x: (burst[x], arrival[x]))
        gantt.append((i + 1, time, time + burst[i]))
        time += burst[i]
        comp[i] = time
        done[i] = True
    return gantt, _build_table(arrival, burst, comp, gantt)


def round_robin(arrival, burst, quantum):
    n = len(arrival)
    remaining = list(burst)
    order = sorted(range(n), key=lambda i: arrival[i])
    queue, time, gantt, comp = [], 0, [], [0] * n
    in_queue = [False] * n
    idx = 0                                   # next process to arrive

    def add_arrived(t):
        nonlocal idx
        while idx < n and arrival[order[idx]] <= t:
            queue.append(order[idx])
            in_queue[order[idx]] = True
            idx += 1

    add_arrived(time)
    while queue or idx < n:
        if not queue:                         # CPU idle
            time = arrival[order[idx]]
            add_arrived(time)
        i = queue.pop(0)
        run = min(quantum, remaining[i])
        gantt.append((i + 1, time, time + run))
        time += run
        remaining[i] -= run
        add_arrived(time)                     # new arrivals go first
        if remaining[i] > 0:
            queue.append(i)
        else:
            comp[i] = time
    return gantt, _build_table(arrival, burst, comp, gantt)


def run_cpu(arrival, burst, quantum):
    results = {}
    for name, fn in [("FCFS", lambda: fcfs(arrival, burst)),
                     ("SJF", lambda: sjf(arrival, burst)),
                     ("Round Robin", lambda: round_robin(arrival, burst, quantum))]:
        gantt, table = fn()
        avg_wt = sum(r["WT"] for r in table.values()) / len(table)
        avg_tat = sum(r["TAT"] for r in table.values()) / len(table)
        avg_rt = sum(r["RT"] for r in table.values()) / len(table)
        results[name] = {"gantt": gantt, "table": table,
                         "avg_wt": round(avg_wt, 2), "avg_tat": round(avg_tat, 2),
                         "avg_rt": round(avg_rt, 2)}
    return results


# ---------------------------------------------------------------
# 2. PAGE REPLACEMENT
# each function returns (faults, steps)
#   steps : list of dict(page, frames, status)
# ---------------------------------------------------------------


def fifo_pr(pages, frames):
    mem, steps, faults = [], [], 0
    for p in pages:
        if p in mem:
            status = "Hit"
        else:
            status = "Fault"
            faults += 1
            if len(mem) == frames:
                mem.pop(0)                    # remove the oldest page
            mem.append(p)
        steps.append({"page": p, "frames": list(mem), "status": status})
    return faults, steps


def lru_pr(pages, frames):
    mem, steps, faults = [], [], 0            # mem[0] = least recently used
    for p in pages:
        if p in mem:
            status = "Hit"
            mem.remove(p)
            mem.append(p)
        else:
            status = "Fault"
            faults += 1
            if len(mem) == frames:
                mem.pop(0)
            mem.append(p)
        steps.append({"page": p, "frames": list(mem), "status": status})
    return faults, steps


def lfu_pr(pages, frames):
    mem, freq, steps, faults = [], {}, [], 0
    for p in pages:
        if p in mem:
            status = "Hit"
            freq[p] += 1
        else:
            status = "Fault"
            faults += 1
            if len(mem) == frames:
                # least frequently used; tie -> the one that came in first
                victim = min(mem, key=lambda x: (freq[x], mem.index(x)))
                mem.remove(victim)
                del freq[victim]
            mem.append(p)
            freq[p] = 1
        steps.append({"page": p, "frames": list(mem), "status": status})
    return faults, steps


def run_page(pages, frames):
    results = {}
    for name, fn in [("FIFO", fifo_pr), ("LRU", lru_pr), ("LFU", lfu_pr)]:
        faults, steps = fn(pages, frames)
        hits = len(pages) - faults
        results[name] = {"faults": faults, "hits": hits, "steps": steps,
                         "hit_ratio": round(hits / len(pages) * 100, 1)}
    return results


# ---------------------------------------------------------------
# 3. DISK SCHEDULING  (new topic)
# each function returns the list of head positions (the path)
# total head movement = sum of |difference| between neighbours
# ---------------------------------------------------------------


def disk_fcfs(requests, head, disk_size):
    return [head] + list(requests)


def disk_sstf(requests, head, disk_size):
    pending, path, cur = list(requests), [head], head
    while pending:
        nxt = min(pending, key=lambda r: abs(r - cur))   # closest request
        pending.remove(nxt)
        path.append(nxt)
        cur = nxt
    return path


def disk_scan(requests, head, disk_size, direction="up"):
    """Elevator algorithm: go to one end, then reverse."""
    lower = sorted(r for r in requests if r < head)
    upper = sorted(r for r in requests if r >= head)
    if direction == "up":
        path = [head] + upper + [disk_size - 1] + lower[::-1]
    else:
        path = [head] + lower[::-1] + [0] + upper
    # remove accidental duplicate end-point (if end already requested)
    clean = [path[0]]
    for p in path[1:]:
        if p != clean[-1]:
            clean.append(p)
    return clean


def disk_cscan(requests, head, disk_size, direction="up"):
    """Circular SCAN: go to one end, jump to the other end, continue."""
    lower = sorted(r for r in requests if r < head)
    upper = sorted(r for r in requests if r >= head)
    if direction == "up":
        path = [head] + upper + [disk_size - 1, 0] + lower
    else:
        path = [head] + lower[::-1] + [0, disk_size - 1] + upper[::-1]
    clean = [path[0]]
    for p in path[1:]:
        if p != clean[-1]:
            clean.append(p)
    return clean


def total_movement(path):
    return sum(abs(path[i + 1] - path[i]) for i in range(len(path) - 1))


def run_disk(requests, head, disk_size, direction):
    results = {}
    funcs = [("FCFS", disk_fcfs), ("SSTF", disk_sstf),
             ("SCAN", lambda r, h, d: disk_scan(r, h, d, direction)),
             ("C-SCAN", lambda r, h, d: disk_cscan(r, h, d, direction))]
    for name, fn in funcs:
        path = fn(requests, head, disk_size)
        results[name] = {"path": path, "movement": total_movement(path)}
    return results


# ---------------------------------------------------------------
# MACHINE LEARNING  (Decision Tree, one model per topic)
# 1) generate random workloads  2) simulate  3) best = label
# 4) train Decision Tree        5) predict for the user's input
# ---------------------------------------------------------------

CPU_LABELS = ["FCFS", "SJF", "Round Robin"]
PAGE_LABELS = ["FIFO", "LRU", "LFU"]
DISK_LABELS = ["FCFS", "SSTF", "SCAN", "C-SCAN"]

CPU_FEATURES = ["Processes", "Mean burst", "Burst spread", "Mean arrival"]
PAGE_FEATURES = ["Ref length", "Unique pages", "Frames", "Unique ratio"]
DISK_FEATURES = ["Requests", "Head position", "Mean request", "Request spread"]


def cpu_features(arrival, burst):
    return [len(burst), float(np.mean(burst)), float(np.std(burst)),
            float(np.mean(arrival))]


def page_features(pages, frames):
    u = len(set(pages))
    return [len(pages), u, frames, u / len(pages)]


def disk_features(requests, head):
    return [len(requests), head, float(np.mean(requests)),
            float(np.std(requests))]


def _best(scores, labels):
    """Pick label with the smallest score (ties -> first in list)."""
    return labels[int(np.argmin([scores[l] for l in labels]))]


def train_cpu_model(n=400, seed=1):
    rnd = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        k = rnd.randint(3, 7)
        arrival = [rnd.randint(0, 8) for _ in range(k)]
        burst = [rnd.randint(1, 15) for _ in range(k)]
        res = run_cpu(arrival, burst, rnd.randint(2, 5))
        X.append(cpu_features(arrival, burst))
        y.append(_best({l: res[l]["avg_wt"] for l in CPU_LABELS}, CPU_LABELS))
    return DecisionTreeClassifier(max_depth=4, random_state=1).fit(X, y)


def train_page_model(n=400, seed=2):
    rnd = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        length = rnd.randint(10, 25)
        top = rnd.randint(4, 9)
        pages = [rnd.randint(0, top) for _ in range(length)]
        frames = rnd.randint(2, 5)
        res = run_page(pages, frames)
        X.append(page_features(pages, frames))
        y.append(_best({l: res[l]["faults"] for l in PAGE_LABELS}, PAGE_LABELS))
    return DecisionTreeClassifier(max_depth=4, random_state=1).fit(X, y)


def train_disk_model(n=400, seed=3):
    rnd = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        k = rnd.randint(4, 10)
        reqs = [rnd.randint(0, 199) for _ in range(k)]
        head = rnd.randint(0, 199)
        res = run_disk(reqs, head, 200, "up")
        X.append(disk_features(reqs, head))
        y.append(_best({l: res[l]["movement"] for l in DISK_LABELS}, DISK_LABELS))
    return DecisionTreeClassifier(max_depth=4, random_state=1).fit(X, y)


def predict_with_confidence(model, features):
    """Return (predicted label, {label: probability %})."""
    probs = model.predict_proba([features])[0]
    chances = {str(c): round(float(p) * 100, 1) for c, p in zip(model.classes_, probs)}
    return str(model.predict([features])[0]), chances
