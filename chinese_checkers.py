#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chinese-checkers 跳棋

双人六角星跳棋：走子 / 连跳，把 10 颗子全部送进对方营地获胜。
纯 Python 标准库，文字棋盘，支持人机对战与 AI 自动对弈。

坐标：axial 六角坐标 (q, r)，六个方向：
  (1,0) (1,-1) (0,-1) (-1,0) (-1,1) (0,1)
"""

import argparse
import random
import sys

DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
SYMBOL = {0: "●", 1: "○"}


def hex_dist(a, b):
    """六角网格曼哈顿距离。"""
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def rot60(cell):
    """axial 坐标逆时针旋转 60 度。"""
    q, r = cell
    return (-r, q + r)


def build_board():
    """返回 (cells, triangles)。

    cells：全部 121 格；triangles：6 个角三角形，每角 10 格。
    triangles[0] 是顶部角（玩家 0 的家），triangles[3] 是底部角。
    """
    cells = set()
    for q in range(-4, 5):
        for r in range(-4, 5):
            if max(abs(q), abs(r), abs(q + r)) <= 4:
                cells.add((q, r))
    top = {(q, r) for r in range(-8, -4) for q in range(0, r + 9)}
    triangles = [top]
    for _ in range(5):
        triangles.append({rot60(c) for c in triangles[-1]})
    for t in triangles:
        cells |= t
    return cells, triangles


class Game:
    def __init__(self, seed=None):
        self.cells, self.triangles = build_board()
        self.home = {0: self.triangles[0], 1: self.triangles[3]}
        self.goal = {0: self.triangles[3], 1: self.triangles[0]}
        self.goal_apex = {0: (0, 8), 1: (0, -8)}
        self.rng = random.Random(seed)
        self.marbles = {}
        for c in self.home[0]:
            self.marbles[c] = 0
        for c in self.home[1]:
            self.marbles[c] = 1
        self.turn = 0
        self.plies = 0
        self.last_move = {0: None, 1: None}  # 每家上一步，用于禁走回头路

    @staticmethod
    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    def legal_moves(self, player):
        """返回 [(起点, 终点)]：一步走子 + 全部连跳终点。"""
        occ = self.marbles
        moves = []
        for s, p in occ.items():
            if p != player:
                continue
            for d in DIRS:
                n = self.add(s, d)
                if n in self.cells and n not in occ:
                    moves.append((s, n))
            seen = {s}
            stack = [s]
            while stack:
                cur = stack.pop()
                for d in DIRS:
                    mid = self.add(cur, d)
                    land = self.add(cur, (2 * d[0], 2 * d[1]))
                    if (mid in occ and land in self.cells
                            and land not in occ and land not in seen):
                        seen.add(land)
                        stack.append(land)
                        moves.append((s, land))
        return moves

    def apply(self, move):
        s, e = move
        player = self.marbles.pop(s)
        self.marbles[e] = player
        self.last_move[player] = move
        self.turn = 1 - self.turn
        self.plies += 1

    def winner(self):
        for p in (0, 1):
            if all(self.marbles.get(c) == p for c in self.goal[p]):
                return p
        return None

    def position_key(self):
        return (tuple(sorted((q, r, p) for (q, r), p in self.marbles.items())),
                self.turn)

    def ai_choose(self, player):
        """贪心 + 小概率随机扰动：以「离最近空格」的距离衡量进度，
        优先推进最落后的子；已进营的子不许出来；禁止立即走回头路
        （破 2-循环），5% 随机走子帮助逃离局部最优。"""
        goal = self.goal[player]
        # 还需要填的格子：目标营里未被我方占据的（开局时是对方的子，会离开）
        empty = [c for c in goal if self.marbles.get(c) != player]
        if not empty:
            return None

        def need(cell):
            return min(hex_dist(cell, e) for e in empty)

        moves = self.legal_moves(player)
        if not moves:
            return None
        lm = self.last_move[player]
        tabu = (lm[1], lm[0]) if lm else None
        cands = [m for m in moves if m != tabu] or moves

        if self.rng.random() < 0.05:
            safe = [m for m in cands
                    if not (m[0] in goal and m[1] not in goal)] or cands
            return self.rng.choice(safe)

        dists = {c: need(c) for c, p in self.marbles.items() if p == player}
        best, best_key = None, None
        for s, e in cands:
            nd = dists.copy()
            del nd[s]
            nd[e] = need(e)
            vals = list(nd.values())
            penalty = 1000 if (s in goal and e not in goal) else 0
            key = (max(vals), sum(vals) + penalty, self.rng.random())
            if best_key is None or key < best_key:
                best, best_key = (s, e), key
        return best

    def render(self):
        rows = {}
        for (q, r) in self.cells:
            rows.setdefault(r, []).append(q)
        width = max(max(qs) - min(qs) for qs in rows.values())
        lines = []
        for r in sorted(rows):
            qs = sorted(rows[r])
            indent = " " * (width - (qs[-1] - qs[0]))
            cells = []
            for q in qs:
                p = self.marbles.get((q, r))
                cells.append(SYMBOL[p] if p is not None else "·")
            lines.append(indent + " ".join(cells))
        return "\n".join(lines)


def play_game(seed=None, ply_cap=600, verbose=False):
    """AI 对 AI 下完一局，返回 0 / 1 / None(和棋)。"""
    g = Game(seed=seed)
    seen = {}
    while g.plies < ply_cap:
        w = g.winner()
        if w is not None:
            if verbose:
                print(g.render())
                print(f"玩家 {w} 获胜！")
            return w
        key = g.position_key()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            return None  # 循环，和棋
        move = g.ai_choose(g.turn)
        if move is None:
            return None
        g.apply(move)
    return None


def auto_play(n_games, seed=None, ply_cap=600):
    rng = random.Random(seed)
    results = {0: 0, 1: 0, None: 0}
    for i in range(n_games):
        w = play_game(seed=rng.randrange(1 << 30), ply_cap=ply_cap)
        results[w] += 1
        tag = "和棋" if w is None else f"玩家 {w} 胜"
        print(f"第 {i + 1}/{n_games} 局：{tag}")
    print(f"总计：玩家 0 胜 {results[0]}，玩家 1 胜 {results[1]}，"
          f"和棋 {results[None]}")
    return results


def interactive(seed=None):
    g = Game(seed=seed)
    print("跳棋：你是 ●（玩家 0），AI 是 ○（玩家 1）")
    print("每轮列出所有合法走法，输入序号走子，q 退出。\n")
    while True:
        print(g.render())
        w = g.winner()
        if w is not None:
            print(f"玩家 {w}（{'你' if w == 0 else 'AI'}）获胜！")
            return
        if g.turn == 0:
            moves = g.legal_moves(0)
            for i, (s, e) in enumerate(moves):
                kind = "走" if hex_dist(s, e) == 1 else "跳"
                print(f"{i:3d}: {s} -> {e} {kind}")
            try:
                raw = input("走哪步？ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                return
            if raw.lower() == "q":
                return
            if not raw.isdigit() or int(raw) >= len(moves):
                print("序号无效，重输。")
                continue
            g.apply(moves[int(raw)])
        else:
            move = g.ai_choose(1)
            if move is None:
                print("AI 无棋可走，和棋。")
                return
            s, e = move
            kind = "走" if hex_dist(s, e) == 1 else "跳"
            print(f"AI：{s} -> {e} {kind}")
            g.apply(move)


def main(argv=None):
    ap = argparse.ArgumentParser(description="跳棋 chinese-checkers")
    ap.add_argument("--auto", type=int, default=0, metavar="N",
                    help="AI 对 AI 自动下 N 局")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    ap.add_argument("--plies", type=int, default=600,
                    help="单局最大手数，超了判和棋")
    args = ap.parse_args(argv)
    if args.auto > 0:
        auto_play(args.auto, seed=args.seed, ply_cap=args.plies)
    else:
        if not sys.stdin.isatty():
            print("交互模式需要终端；用 --auto N 看 AI 对弈。", file=sys.stderr)
            sys.exit(2)
        interactive(seed=args.seed)


if __name__ == "__main__":
    main()
