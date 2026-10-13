"""
生成贴近真实 VOC 分布的随机标注数据集。

用法:
    python -m scripts.generate_dataset --n 500 --seed 42 --output data/voc_dataset.jsonl

输出 JSONL 字段: id / text / source / product_line / product / category / sub_category
标签对齐到 src.state 的枚举体系 (FeedbackCategory / FeedbackSubCategory / XiaomiProductLine)。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# (product_line, [具体产品名], weight)
PRODUCT_LINES: list[tuple[str, list[str], float]] = [
    ("手机", ["小米14", "小米14 Pro", "Redmi K70", "Redmi Note 13", "小米MIX Flip"], 0.30),
    ("电视", ["小米电视 S Pro", "Redmi 电视 A Pro", "小米电视大师版"], 0.10),
    ("笔记本", ["RedmiBook Pro", "小米笔记本 Air", "Redmi G 游戏本"], 0.08),
    ("平板", ["小米平板 6 Pro", "Redmi Pad"], 0.06),
    ("智能穿戴", ["小米手环 9", "Redmi Watch 4", "小米手表 S4"], 0.08),
    ("智能家居", ["智能门锁", "扫地机器人", "米家摄像头", "小爱音箱", "智能灯"], 0.14),
    ("IoT生态链", ["米家台灯", "智能插座", "空气净化器"], 0.06),
    ("大家电", ["小米空调", "小米冰箱", "小米洗衣机"], 0.06),
    ("路由器", ["小米路由器 AX6000", "Redmi 路由器"], 0.04),
    ("配件", ["移动电源", "充电器", "蓝牙耳机", "数据线"], 0.05),
    ("小米汽车", ["小米SU7", "SU7 Max", "SU7 Pro"], 0.03),
]

SOURCES = ["社区", "商城", "服务频道"]
SOURCE_WEIGHTS = [0.6, 0.25, 0.15]

FEATS = ["分屏多任务", "视频倍速播放", "息屏显示", "长截图", "文件互传",
         "儿童模式", "通话录音", "应用双开", "隐私水印", "AI字幕"]
UI_FEATS = ["暗黑模式", "桌面布局", "图标主题", "字体大小", "下拉通知栏",
            "过场动效", "负一屏", "控制中心"]

# sub = (category, sub_category, weight, templates, feats)
SUBCATS: list[dict] = [
    {
        "category": "售后报修", "sub_category": "软件缺陷", "weight": 0.17, "feats": None,
        "templates": [
            "{product}老是闪退，一打开就崩溃",
            "{product}系统卡顿严重，反应特别慢",
            "{product}升级新系统后出现各种bug",
            "用{product}的APP经常黑屏死机",
            "更新后{product}直接报错打不开",
            "{product}系统异常，经常自动重启",
            "{product}这个系统真的不好用，总出毛病",
        ],
    },
    {
        "category": "售后报修", "sub_category": "硬件故障", "weight": 0.13, "feats": None,
        "templates": [
            "{product}无法开机了",
            "{product}充电充不进去",
            "{product}屏幕碎了，需要换屏",
            "{product}电池鼓包了",
            "{product}按键失灵了",
            "{product}突然黑屏，疑似硬件问题",
            "{product}指纹识别经常失败",
        ],
    },
    {
        "category": "售后报修", "sub_category": "物流问题", "weight": 0.08, "feats": None,
        "templates": [
            "{product}下单一周了还没发货",
            "物流太慢，{product}迟迟不送到",
            "快递显示签收但没收到{product}",
            "{product}发错地址了",
            "{product}退货物流一直没更新",
        ],
    },
    {
        "category": "舆情吐槽", "sub_category": "产品质量", "weight": 0.14, "feats": None,
        "templates": [
            "{product}质量太差了，没用几天就坏了",
            "做工粗糙，{product}有明显瑕疵",
            "品控堪忧，{product}有质量问题",
            "{product}用了三天就出问题了",
            "{product}太容易坏了，很失望",
        ],
    },
    {
        "category": "舆情吐槽", "sub_category": "服务态度", "weight": 0.12, "feats": None,
        "templates": [
            "{product}的客服态度非常恶劣",
            "投诉了根本没人处理",
            "{product}售后服务太差劲了",
            "客服电话一直打不通",
            "售后推诿扯皮，不解决{product}的问题",
        ],
    },
    {
        "category": "舆情吐槽", "sub_category": "价格问题", "weight": 0.06, "feats": None,
        "templates": [
            "{product}价格太贵了",
            "刚买{product}就降价，太坑了",
            "{product}性价比太低，不值这个价",
            "{product}悄悄涨价，吃相难看",
        ],
    },
    {
        "category": "作品建议", "sub_category": "功能建议", "weight": 0.12, "feats": FEATS,
        "templates": [
            "建议{product}增加{feat}功能",
            "希望{product}支持{feat}",
            "能不能给{product}加个{feat}",
            "{product}要是能有{feat}就好了",
            "期待{product}加入{feat}",
        ],
    },
    {
        "category": "作品建议", "sub_category": "界面优化", "weight": 0.09, "feats": UI_FEATS,
        "templates": [
            "建议{product}增加{feat}",
            "{product}的界面太乱了，希望优化",
            "希望优化{product}的UI布局",
            "{product}图标和配色不好看",
            "{product}的{feat}做得很差，赶紧改",
        ],
    },
    {
        "category": "作品建议", "sub_category": "内容推荐", "weight": 0.09, "feats": None,
        "templates": [
            "{product}推荐算法很不准，总推不感兴趣的",
            "希望优化{product}的推荐内容",
            "{product}推送太频繁了，很烦",
            "{product}三天两头给我推广告",
        ],
    },
]

PREFIXES = ["", "", "说真的，", "说实话，", "请问下，", "想问下，", "大家有没有遇到，"]
SUFFIXES = ["", "", "，真的无语", "，太失望了", "，服了", "，气死我了", "，谁来管管", ""]
EMOJIS = ["", "", "", "😭", "😡", "🤔", "🙏", "💢"]


def _typo(text: str, rng: random.Random) -> str:
    if rng.random() > 0.04 or not text:
        return text
    i = rng.randrange(len(text))
    return text[:i] + text[i] + text[i:]


def _add_noise(text: str, rng: random.Random) -> str:
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    emoji = rng.choice(EMOJIS)
    return _typo(prefix + text + suffix + emoji, rng)


def generate_dataset(n: int = 500, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for idx in range(1, n + 1):
        sub = rng.choices(SUBCATS, weights=[s["weight"] for s in SUBCATS])[0]
        line, products, _ = rng.choices(
            PRODUCT_LINES, weights=[p[2] for p in PRODUCT_LINES]
        )[0]
        product = rng.choice(products)
        template = rng.choice(sub["templates"])
        feat = rng.choice(sub["feats"]) if sub["feats"] else ""
        text = template.format(product=product, feat=feat)
        text = _add_noise(text, rng)
        records.append({
            "id": f"VOC-{idx:06d}",
            "text": text,
            "source": rng.choices(SOURCES, weights=SOURCE_WEIGHTS)[0],
            "product_line": line,
            "product": product,
            "category": sub["category"],
            "sub_category": sub["sub_category"],
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 VOC 随机标注数据集")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="data/voc_dataset.jsonl")
    args = ap.parse_args()

    records = generate_dataset(args.n, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_sub = Counter(r["sub_category"] for r in records)
    by_cat = Counter(r["category"] for r in records)
    print(f"已生成 {len(records)} 条 -> {out}")
    print("按一级分类:", dict(by_cat))
    print("按二级分类:", dict(by_sub))


if __name__ == "__main__":
    main()