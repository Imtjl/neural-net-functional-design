#!/usr/bin/env python3
"""Draws the report figures for OVS lab 1 from the output of build/nn.

Run from the repo root after `make`:  python3 tools/report_figures.py
Every number in the tables comes from running the program on this machine.
Output: report/figures/*.png
"""

import pathlib
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = pathlib.Path("report/figures")
BIN = "./build/nn"
BEST = "49-8-3"
BEST_ALPHA = "0.30"
BEST_EPS = "0.1"
EPS_VALUES = ["0.1", "0.05", "0.01", "0.001"]

COLORS = {"circle": "#2f6fdf", "square": "#2e9e5b",
          "triangle": "#e08a1e", "noise": "#7a7a7a"}
RU = {"circle": "круг", "square": "квадрат",
      "triangle": "треугольник", "noise": "шум"}
FAIL_COLOR = "#d62828"
HIGHLIGHT = "#fff2a8"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def step(message):
    print(f"[figures] {message}")


def run(*args):
    result = subprocess.run([BIN, *args], capture_output=True, text=True,
                            check=True)
    return result.stdout


def load_samples(path):
    samples = []
    for block in pathlib.Path(path).read_text().strip().split("\n\n"):
        lines = block.strip().split("\n")
        samples.append({"label": lines[0].split()[0], "name": lines[0],
                        "grid": lines[1:8]})
    return samples


def parse_demo(text):
    predictions = {}
    pattern = r"^([\d.]+) ([\d.]+) ([\d.]+)\s+(OK|FAIL)\s+(.+?) -> (\w+)$"
    for m in re.finditer(pattern, text, re.M):
        predictions[m.group(5)] = {"ok": m.group(4) == "OK",
                                   "answer": m.group(6)}
    metrics = {
        "epochs": re.search(r"training: converged \w+, (\d+) epochs", text)[1],
        "train_ms": re.search(r"epochs, ([\d.]+) ms", text)[1],
        "forward_us": re.search(r"forward pass: ([\d.]+) us", text)[1],
        "weights": re.search(r"\((\d+) weights\)", text)[1],
        "mem_inference": re.search(r"inference:\s+(\d+) B", text)[1],
        "mem_training": re.search(r"training:\s+(\d+) B", text)[1],
        "mem_set": re.search(r"training set:\s+(\d+) B", text)[1],
        "test": re.search(r"test: (\d+/\d+) correct", text)[1],
    }
    return predictions, metrics


def parse_table(text):
    pattern = (r"^(\S+)\s+(\d+)\s+(\d+)/(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
               r"([\d.]+)/(\d+)\s+([\d.]+)/(\d+) \((\d+)\.\.(\d+)\)$")
    rows = []
    for m in re.finditer(pattern, text, re.M):
        g = m.groups()
        rows.append({"name": g[0], "weights": g[1],
                     "converged": f"{g[2]}/{g[3]}", "epochs": g[4],
                     "train_ms": g[5], "train_ok": f"{float(g[6]):.0f}/{g[7]}",
                     "test_ok": f"{g[8]}/{g[9]}",
                     "test_range": f"{g[10]}..{g[11]}"})
    return rows


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    step(f"saved {OUT / filename}")


def draw_image(ax, x, y, size, grid, color, border=None):
    """Draws one 7x7 image with its top-left corner at (x, y)."""
    cell = size / 7
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            ax.add_patch(Rectangle((x + c * cell, y - (r + 1) * cell), cell,
                                   cell, facecolor=color if ch == "#" else
                                   "white", edgecolor="#d0d0d0", lw=0.4))
    if border:
        ax.add_patch(Rectangle((x - 0.04 * size, y - 1.04 * size),
                               1.08 * size, 1.08 * size, fill=False,
                               edgecolor=border, lw=2.5))


def draw_network(sample):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(-0.5, 16.8)
    ax.set_ylim(-3.6, 11.2)
    ax.axis("off")

    xs = {"in": 4.0, "hid": 8.0, "out": 11.5}
    pixels = [ch for row in sample["grid"] for ch in row]
    y_in = [10 - i * 10 / 48 for i in range(49)]
    y_hid = [9 - i * 8 / 7 for i in range(8)]
    y_out = [7, 5, 3]
    classes = ["circle", "square", "triangle"]

    draw_image(ax, 0, 7.4, 2.6, sample["grid"], COLORS["circle"])
    ax.text(1.3, 4.3, "картинка 7×7\n→ 49 входов\nпострочно",
            ha="center", va="top", fontsize=10)
    ax.add_patch(FancyArrowPatch((2.8, 6.1), (3.7, 6.1),
                                 arrowstyle="-|>", mutation_scale=18))

    for yi in y_in:
        for yh in y_hid:
            ax.plot([xs["in"], xs["hid"]], [yi, yh], color="#9aa5b1",
                    lw=0.25, alpha=0.5, zorder=1)
    for yh in y_hid:
        for yo in y_out:
            ax.plot([xs["hid"], xs["out"]], [yh, yo], color="#5a6570",
                    lw=0.8, alpha=0.8, zorder=1)

    for yi, p in zip(y_in, pixels):
        ax.scatter(xs["in"], yi, s=28, zorder=3,
                   color="black" if p == "#" else "white",
                   edgecolors="black", linewidths=0.6)
    for yh in y_hid:
        ax.scatter(xs["hid"], yh, s=420, zorder=3, color="#dbe4f0",
                   edgecolors="black", linewidths=1)
    for yo, cls in zip(y_out, classes):
        ax.scatter(xs["out"], yo, s=520, zorder=3, color=COLORS[cls],
                   edgecolors="black", linewidths=1)
        ax.text(xs["out"] + 0.45, yo, RU[cls], va="center", fontsize=12)

    ax.text(xs["in"], 10.7, "Входы\n49 пикселей, k = 0", ha="center",
            fontsize=11)
    ax.text(xs["hid"], 10.0, "Скрытый слой\n8 нейронов, k = 1", ha="center",
            fontsize=11)
    ax.text(xs["out"], 8.3, "Выходной слой\n3 нейрона, k = 2", ha="center",
            fontsize=11)
    ax.text(6.0, 0.35, r"$w^1$: 8 × 49 = 392 веса", ha="center",
            fontsize=11, bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(9.75, 1.6, r"$w^2$: 3 × 8 = 24 веса", ha="center", fontsize=11)

    ax.add_patch(FancyBboxPatch((14.3, 4.0), 2.2, 2.0,
                                boxstyle="round,pad=0.1", fc="#f2f2f2",
                                ec=COLORS["noise"]))
    ax.text(15.4, 5.0, "ответ = argmax\nесли max $y_i$ < 0.5:\nшум",
            ha="center", va="center", fontsize=10.5)

    ax.add_patch(FancyArrowPatch((3.5, -1.3), (12.0, -1.3),
                                 arrowstyle="-|>", mutation_scale=22,
                                 color="#1b4f9c", lw=2))
    ax.text(7.75, -1.1, r"Прямой проход:  $y_i^k = f\left(\sum_j y_j^{k-1}"
            r"w_{ij}^k\right)$,   $f(x) = \dfrac{1}{1 + e^{-x}}$",
            ha="center", va="bottom", fontsize=12.5, color="#1b4f9c")

    ax.add_patch(FancyArrowPatch((12.0, -2.2), (3.5, -2.2),
                                 arrowstyle="-|>", mutation_scale=22,
                                 color="#a1301d", lw=2))
    ax.text(7.75, -2.4, "Обратный проход:  "
            r"$\delta_i^2 = y_i^2(1 - y_i^2)(t_i - y_i^2)$,   "
            r"$\delta_i^1 = y_i^1(1 - y_i^1)\sum_j \delta_j^2 w_{ji}^2$,   "
            r"$\Delta w_{ij}^k = \alpha\,\delta_i^k\,y_j^{k-1}$",
            ha="center", va="top", fontsize=12, color="#a1301d")
    ax.text(15.4, 2.6, r"$E = \frac{1}{2}\sum_i (t_i - y_i^2)^2$",
            ha="center", fontsize=12.5)
    ax.text(15.4, 1.9, "цель t: (1,0,0) круг,\n(0,1,0) квадрат,\n"
            "(0,0,1) треугольник,\n(0,0,0) шум", ha="center", va="top",
            fontsize=9.5)
    save(fig, "01_network_49-8-3.png")


def draw_training_loop():
    fig, ax = plt.subplots(figsize=(10, 10.5))
    ax.set_xlim(-0.8, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(x, y, text, color="#dbe4f0", w=4.6, h=0.9):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.08", fc=color,
                                    ec="black"))
        ax.text(x, y, text, ha="center", va="center", fontsize=11)

    def arrow(p, q, label=None, label_pos=None):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>",
                                     mutation_scale=16, lw=1.3))
        if label:
            ax.text(*label_pos, label, fontsize=10.5, color="#444")

    box(4, 10.3, "Начальные веса: случайные из [−0.5, 0.5], seed",
        w=5.8)
    box(4, 9.0, "Новая эпоха: по порядку каждая картинка\nобучающей "
        "выборки (16 шт.)")
    box(4, 7.6, r"Прямой проход: $y = $ forward(пиксели)")
    box(4, 6.3, r"Ошибка: $E = \frac{1}{2}\sum_i (t_i - y_i)^2$")
    box(4, 5.0, r"$E$ > eps (0.1) ?", color="#fff2a8", w=3.2)
    box(8.2, 5.0, "Обратный проход:\nδ всех слоёв,\n"
        r"$\Delta w = \alpha\,\delta\,y$", color="#f6d5cc", w=2.6,
        h=1.5)
    box(4, 3.6, "Следующая картинка;\nпосле последней: конец эпохи")
    box(4, 2.2, "Во всей эпохе ни одного обратного прохода?",
        color="#fff2a8", w=5.4)
    box(4, 0.8, "Стоп: сеть обучена", color="#cfe8d5", w=3.2)

    arrow((4, 9.85), (4, 9.5))
    arrow((4, 8.5), (4, 8.05))
    arrow((4, 7.15), (4, 6.75))
    arrow((4, 5.85), (4, 5.45))
    arrow((5.6, 5.0), (6.9, 5.0), "да", (6.0, 5.15))
    arrow((4, 4.55), (4, 4.05), "нет", (4.15, 4.2))
    arrow((8.2, 4.25), (6.3, 3.75))
    ax.plot([1.7, 1.0, 1.0], [3.6, 3.6, 7.6], color="black", lw=1.3)
    arrow((1.0, 7.6), (1.7, 7.6))
    ax.text(0.85, 5.6, "ещё есть\nкартинки", ha="right", fontsize=10.5,
            color="#444")
    arrow((4, 3.15), (4, 2.65))
    arrow((4, 1.75), (4, 1.25), "да", (4.15, 1.4))
    ax.plot([6.7, 9.6, 9.6], [2.2, 2.2, 9.0], color="black", lw=1.3)
    arrow((9.6, 9.0), (6.3, 9.0), "нет", (8.8, 2.35))
    save(fig, "02_training_loop.png")


def draw_collage(samples, title, filename, predictions=None, per_row=None):
    classes = ["circle", "square", "triangle", "noise"]
    groups = [[s for s in samples if s["label"] == c] for c in classes]
    per_row = per_row or max(len(g) for g in groups)
    size, gap_x, gap_y = 1.0, 0.45, 0.95
    width = per_row * (size + gap_x) + 1.6
    height = len(groups) * (size + gap_y)
    fig, ax = plt.subplots(figsize=(width * 1.25, height * 1.25 + 0.6))
    ax.set_xlim(-1.7, width - 1.6)
    ax.set_ylim(-0.2, height + 0.1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, loc="left")

    for gi, (cls, group) in enumerate(zip(classes, groups)):
        top = height - gi * (size + gap_y)
        ax.text(-0.25, top - size / 2, RU[cls], ha="right", va="center",
                fontsize=12, color=COLORS[cls], weight="bold")
        for i, s in enumerate(group):
            x = i * (size + gap_x)
            tag = s["name"].split(" ", 1)[1]
            border, caption, caption_color = None, tag, "#333"
            if predictions is not None:
                p = predictions[s["name"]]
                if not p["ok"]:
                    border = FAIL_COLOR
                    caption = f"{tag}\n→ {RU[p['answer']]}"
                    caption_color = FAIL_COLOR
            draw_image(ax, x, top, size, s["grid"], COLORS[cls], border)
            ax.text(x + size / 2, top - size - 0.08, caption, ha="center",
                    va="top", fontsize=7.5, color=caption_color)
    save(fig, filename)


def draw_table(header, rows, highlight, title, filename):
    fig, ax = plt.subplots(figsize=(1.55 * len(header), 0.45 * len(rows)
                                    + 1.0))
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, loc="left")
    table = ax.table(cellText=rows, colLabels=header, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.6)
    table.auto_set_column_width(list(range(len(header))))
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        if r == 0:
            cell.set_facecolor("#e6e9ef")
            cell.set_text_props(weight="bold")
        elif r - 1 in highlight:
            cell.set_facecolor(HIGHLIGHT)
            cell.set_text_props(weight="bold")
    save(fig, filename)


def test_groups(test, predictions):
    groups = {"Локальные искажения 1–3 пикселя (вкл. рис. 2)": [],
              "Сдвиги на 1 пиксель": [], "Шум": []}
    for s in test:
        key = ("Шум" if s["label"] == "noise" else "Сдвиги на 1 пиксель"
               if "shift" in s["name"] else
               "Локальные искажения 1–3 пикселя (вкл. рис. 2)")
        groups[key].append(predictions[s["name"]]["ok"])
    return [[k, str(len(v)), str(sum(v))] for k, v in groups.items()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_samples("data/train.txt")
    test = load_samples("data/test.txt")
    step(f"loaded {len(train)} train and {len(test)} test images")

    predictions, metrics = parse_demo(run("demo", BEST))
    step(f"demo {BEST}: test {metrics['test']}")
    structures = parse_table(run("compare"))
    step(f"compare: {len(structures)} structures")
    alphas = parse_table(run("alpha", BEST))
    step(f"alpha: {len(alphas)} values")
    eps_rows = []
    for eps in EPS_VALUES:
        row = next(r for r in parse_table(run("compare", eps))
                   if r["name"] == BEST)
        eps_rows.append({**row, "name": eps})
        step(f"compare eps {eps}: done")

    draw_network(train[0])
    draw_training_loop()
    draw_collage(train, "Обучающая выборка: 16 картинок, по 4 на класс",
                 "03_train_set.png")
    draw_collage(test, f"Тестовая выборка: 36 картинок; сеть {BEST}, "
                 f"seed 1: {metrics['test']} верно; красная рамка = ошибка",
                 "04_test_set.png", predictions)

    draw_table(
        ["Структура", "Весов", "Сошлось", "Эпох", "Обучение, мс", "Train",
         "Test (среднее)", "Test (мин..макс)"],
        [[r["name"], r["weights"], r["converged"], r["epochs"],
          r["train_ms"], r["train_ok"], r["test_ok"], r["test_range"]]
         for r in structures],
        [i for i, r in enumerate(structures) if r["name"] == BEST],
        "Сравнение структур: α = 0.3, eps = 0.1, среднее по 10 seed",
        "05_table_structures.png")
    draw_table(
        ["α", "Сошлось", "Эпох", "Обучение, мс", "Test (среднее)",
         "Test (мин..макс)"],
        [[r["name"], r["converged"], r["epochs"], r["train_ms"],
          r["test_ok"], r["test_range"]] for r in alphas],
        [i for i, r in enumerate(alphas) if r["name"] == BEST_ALPHA],
        f"Варьирование α: сеть {BEST}, eps = 0.1, среднее по 10 seed",
        "06_table_alpha.png")
    draw_table(
        ["eps", "Эпох", "Обучение, мс", "Test (среднее)", "Test (мин..макс)"],
        [[r["name"], r["epochs"], r["train_ms"], r["test_ok"],
          r["test_range"]] for r in eps_rows],
        [i for i, r in enumerate(eps_rows) if r["name"] == BEST_EPS],
        f"Длительность обучения (порог eps): сеть {BEST}, α = 0.3",
        "07_table_eps.png")
    draw_table(
        ["Характеристика", "Значение"],
        [["Структура", f"{BEST} ({metrics['weights']} весов)"],
         ["Эпох обучения", metrics["epochs"]],
         ["Время обучения", f"{metrics['train_ms']} мс"],
         ["Время вычисления выхода", f"{metrics['forward_us']} мкс"],
         ["Память: вычисление выхода", f"{metrics['mem_inference']} Б"],
         ["Память: обучение", f"{metrics['mem_training']} Б"],
         ["Память: обучающая выборка", f"{metrics['mem_set']} Б"],
         ["Верно на тесте", metrics["test"]]],
        [],
        f"Выбранная сеть {BEST}: α = 0.3, eps = 0.1, seed 1",
        "08_table_best.png")
    draw_table(
        ["Группа тестовых картинок", "Всего", "Распознано"],
        test_groups(test, predictions), [],
        f"Результат {BEST} на тесте по группам", "09_table_test_groups.png")
    step("all figures done")


if __name__ == "__main__":
    main()
