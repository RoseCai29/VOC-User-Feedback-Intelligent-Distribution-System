"""
生成贴近真实 VOC 分布的随机标注数据集。

用法:
    python -m scripts.generate_dataset --n 3600 --seed 42 --output data/voc_dataset.jsonl

输出 JSONL 字段: id / text / source / product_line / product / category / sub_category
标签对齐到 src.state 的枚举体系 (FeedbackCategory / FeedbackSubCategory / XiaomiProductLine)。

说明: 合成数据只用于冷启动与流程验证，指标不能替代真实标注集的结论。
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
         "儿童模式", "通话录音", "应用双开", "隐私水印", "AI字幕", "通话降噪",
         "全局搜索", "快捷手势", "无线投屏", "多设备协同"]
UI_FEATS = ["暗黑模式", "桌面布局", "图标主题", "字体大小", "下拉通知栏",
            "过场动效", "负一屏", "控制中心", "设置菜单", "通知样式"]

# sub = (category, sub_category, weight, templates, feats)
SUBCATS: list[dict] = [
    {
        "category": "售后报修", "sub_category": "软件缺陷", "weight": 0.12, "feats": None,
        "templates": [
            "{product}老是闪退，一打开就崩溃",
            "{product}系统卡顿严重，点一下要等半天",
            "升级新系统后{product}各种报错",
            "{product}的APP经常黑屏死机",
            "更新完直接打不开，一直转圈",
            "{product}用着用着就自动重启，系统有bug",
            "{product}最近特别卡，动不动就无响应",
            "打开{product}就提示异常，根本用不了",
            "{product}升级后相机打不开了，一直闪退",
            "{product}的APP频繁崩溃，重装也没用",
            "刚更新系统，{product}就开始各种出毛病",
            "{product}死机了三次，系统稳定性太差",
            "看视频的时候{product}突然卡死",
            "{product}通知栏老是卡住，点不动",
            "系统更新后{product}耗电异常，还经常发热重启",
        ],
    },
    {
        "category": "售后报修", "sub_category": "硬件故障", "weight": 0.12, "feats": None,
        "templates": [
            "{product}开不了机，按电源没反应",
            "{product}充电充不进去，换线也没用",
            "{product}屏幕摔碎了，需要换屏",
            "{product}电池鼓包了，后盖都顶起来了",
            "{product}音量键按下去没反应",
            "{product}突然黑屏，怀疑是硬件坏了",
            "{product}指纹识别一直失败，录不进去",
            "{product}充电口接触不良，要来回插好几次",
            "{product}摄像头进灰了，拍照有黑点",
            "{product}扬声器没声音，外放也听不到",
            "{product}开机一直停在logo界面",
            "{product}屏幕出现竖线，越用越多",
            "{product}震动马达坏了，来电没感觉",
            "{product}充到80%就不动了，电池有问题",
            "{product}进水后无法开机，需要检测维修",
        ],
    },
    {
        "category": "售后报修", "sub_category": "物流问题", "weight": 0.10, "feats": None,
        "templates": [
            "{product}下单一周了还没发货",
            "物流太慢，{product}一直不送到",
            "快递显示签收但根本没收到{product}",
            "{product}发错地址了，联系客服也没解决",
            "{product}退货物流一周没更新",
            "买的{product}卡在转运中心三天了",
            "说好次日达，{product}等了五天",
            "{product}显示已发货，但查不到物流信息",
            "快递员直接把{product}放门口，丢了",
            "{product}配送延迟，客服只会让我等",
            "下单十天了{product}还没出库",
            "同一个订单{product}只发了一半",
        ],
    },
    {
        "category": "舆情吐槽", "sub_category": "产品质量", "weight": 0.12, "feats": None,
        "templates": [
            "{product}质量太差了，没用几天就坏",
            "做工粗糙，{product}有明显瑕疵",
            "品控堪忧，{product}缝隙很大",
            "{product}用了三天就出问题，很失望",
            "{product}太容易坏了，感觉是次品",
            "刚拆封{product}就有划痕，品控不行",
            "{product}材质廉价，完全不像这个价位",
            "买了两个{product}都有问题，良品率太低",
            "{product}用一个月就松动，质量堪忧",
            "这做工也太糙了，{product}边角都割手",
            "{product}的品控越来越差，不敢再买",
            "新买的{product}就有异响，质量不行",
        ],
    },
    {
        "category": "舆情吐槽", "sub_category": "服务态度", "weight": 0.12, "feats": None,
        "templates": [
            "{product}的客服态度非常恶劣",
            "投诉了根本没人处理",
            "{product}售后太差，一直在踢皮球",
            "客服电话打了十几次都打不通",
            "售后推诿，说不是他们的责任",
            "问客服问题，直接被挂电话",
            "{product}的售后只会复制粘贴回复",
            "门店店员爱答不理，体验极差",
            "申请售后三天没人联系我",
            "客服答应回电，结果一直没消息",
            "{product}的客服全程敷衍，气得想退货",
            "投诉到平台也没人管，服务太差",
        ],
    },
    {
        "category": "舆情吐槽", "sub_category": "价格问题", "weight": 0.09, "feats": None,
        "templates": [
            "{product}价格太贵了，性价比低",
            "刚买{product}就降价，太坑了",
            "{product}不值这个价",
            "{product}悄悄涨价，吃相难看",
            "同样的{product}别人买便宜五百",
            "{product}首发价太高，果断劝退",
            "会员还要额外收费，价格不合理",
            "{product}维修报价离谱，简直是抢钱",
            "刚下单就出优惠券，价格保护呢",
            "{product}比同配置竞品贵太多",
            "涨价也太突然了，老用户不配吗",
            "{product}的配件卖这么贵，不合理",
        ],
    },
    {
        "category": "作品建议", "sub_category": "功能建议", "weight": 0.12, "feats": FEATS,
        "templates": [
            "建议{product}增加{feat}",
            "希望{product}支持{feat}",
            "能不能给{product}加个{feat}",
            "{product}要是有{feat}就好了",
            "期待{product}加入{feat}功能",
            "强烈建议给{product}做{feat}",
            "{product}缺一个{feat}，很不方便",
            "希望下个版本能上{feat}",
            "{product}的{feat}能不能开放给老机型",
            "建议参考友商的{feat}",
            "加个{feat}吧，这个真的很实用",
            "如果{product}能支持{feat}，体验会好很多",
        ],
    },
    {
        "category": "作品建议", "sub_category": "界面优化", "weight": 0.11, "feats": UI_FEATS,
        "templates": [
            "建议{product}增加{feat}",
            "{product}的界面太乱，希望优化",
            "希望优化{product}的UI布局",
            "{product}图标和配色不好看",
            "{product}的{feat}做得太差，赶紧改",
            "{product}设置层级太深，找不到功能",
            "界面字体太小，老人看不清",
            "{product}动画太多，看着头晕",
            "希望{product}可以自定义桌面布局",
            "{product}的{feat}交互很反人类",
            "页面信息太挤，建议重新设计",
            "{product}深色模式颜色发灰，不好看",
        ],
    },
    {
        "category": "作品建议", "sub_category": "内容推荐", "weight": 0.10, "feats": None,
        "templates": [
            "{product}推荐算法不准，总推不感兴趣的",
            "希望优化{product}的推荐内容",
            "{product}推送太频繁，很烦",
            "{product}三天两头给我推广告",
            "{product}首页全是重复内容，算法有问题",
            "刚聊完就推相关广告，隐私呢",
            "{product}推荐的内容质量越来越差",
            "希望{product}能减少低质营销推送",
            "{product}总推已经看过的内容",
            "{product}的推荐完全不懂我",
            "关闭个性化推荐后还是乱推",
            "{product}信息流广告太多，影响体验",
        ],
    },
]

PREFIXES = ["", "", "", "说真的，", "说实话，", "不是我说，", "请问一下，",
            "有没有人跟我一样，", "刚遇到个事，", "吐槽一下，", "问下大家，"]
SUFFIXES = ["", "", "", "，真的无语", "，太失望了", "，服了", "，气死我了",
            "，谁来管管", "，希望官方看看", "，已经反馈过没用", "，谢谢"]
DETAILS = ["", "", "", "，已经影响到正常使用了", "，希望能尽快解决",
           "，刚买没多久就这样", "，实在没办法了"]
EMOJIS = ["", "", "", "", "😭", "😡", "🤔", "🙏", "💢"]


def _typo(text: str, rng: random.Random) -> str:
    """少量错别字/手滑噪声：重复、漏字、相邻字符互换。"""
    if rng.random() > 0.10 or len(text) < 2:
        return text
    i = rng.randrange(len(text))
    op = rng.randrange(3)
    if op == 0:
        return text[:i] + text[i] + text[i:]
    if op == 1:
        return text[:i] + text[i + 1:]
    if i < len(text) - 1:
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text


def _add_noise(text: str, rng: random.Random) -> str:
    text = rng.choice(PREFIXES) + text + rng.choice(DETAILS)
    text = text + rng.choice(SUFFIXES) + rng.choice(EMOJIS)
    text = _typo(text, rng)
    if rng.random() < 0.15:
        text = text.replace("，", rng.choice([" ", "！", "，"]))
    return text


def generate_dataset(n: int = 3600, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for idx in range(1, n + 1):
        sub = rng.choices(SUBCATS, weights=[s["weight"] for s in SUBCATS])[0]
        line, products, _ = rng.choices(
            PRODUCT_LINES, weights=[p[2] for p in PRODUCT_LINES]
        )[0]
        product = rng.choice(products)
        t_idx = rng.randrange(len(sub["templates"]))
        template = sub["templates"][t_idx]
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
            "template_id": f"{sub['sub_category']}#{t_idx}",
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 VOC 随机标注数据集")
    ap.add_argument("--n", type=int, default=3600)
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
