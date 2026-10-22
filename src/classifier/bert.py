"""
BERT 小模型 —— 共享编码器 + 双分类头。

类别数由分类体系决定（当前 4 类 / 11 类），并把标签列表写入模型目录，
加载旧模型时按保存时的标签列表还原，避免分类体系扩展后无法加载。
与 LinearSmallModel 实现同一 SmallModel 协议，级联路由与 fast path 无需改动。
默认主干使用 hfl/rbt3（3 层中文 RoBERTa），CPU 可训练、可推理；也可通过
--backbone 换成 bert-base-chinese 等更大模型。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from ..state import FeedbackCategory, FeedbackSubCategory
from .base import SmallModelPrediction

logger = logging.getLogger(__name__)

CATEGORIES = [c.value for c in FeedbackCategory]
SUB_CATEGORIES = [s.value for s in FeedbackSubCategory]


class _Heads(nn.Module):
    """共享编码器输出上的两个线性分类头。"""

    def __init__(self, hidden_size: int, n_category: int, n_sub: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.category = nn.Linear(hidden_size, n_category)
        self.sub = nn.Linear(hidden_size, n_sub)


class BertSmallModel:
    """多任务中文 BERT 分类器，实现 SmallModel 协议。"""

    def __init__(
        self,
        backbone,
        tokenizer,
        heads: _Heads,
        max_len: int = 64,
        backbone_name: str = "",
        device: str | None = None,
        categories: list[str] | None = None,
        sub_categories: list[str] | None = None,
    ):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.heads = heads
        self.max_len = max_len
        self.backbone_name = backbone_name
        self.categories = list(categories or CATEGORIES)
        self.sub_categories = list(sub_categories or SUB_CATEGORIES)
        self.device = torch.device(device or "cpu")
        self.backbone.to(self.device)
        self.heads.to(self.device)
        self.backbone.eval()
        self.heads.eval()

    def is_ready(self) -> bool:
        return self.backbone is not None and self.tokenizer is not None and self.heads is not None

    def predict(self, text: str) -> SmallModelPrediction | None:
        if not text.strip() or not self.is_ready():
            return None
        try:
            enc = self.tokenizer(
                [text], truncation=True, max_length=self.max_len,
                padding=True, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                pooled = self.backbone(**enc).last_hidden_state[:, 0]
                pooled = self.heads.dropout(pooled)
                cat_probs = torch.softmax(self.heads.category(pooled), dim=-1)[0]
                sub_probs = torch.softmax(self.heads.sub(pooled), dim=-1)[0]
        except Exception as e:
            logger.warning("BERT 推理失败: %s", e)
            return None

        ci = int(cat_probs.argmax())
        si = int(sub_probs.argmax())
        return SmallModelPrediction(
            category=self.categories[ci],
            sub_category=self.sub_categories[si],
            category_confidence=float(cat_probs[ci]),
            sub_confidence=float(sub_probs[si]),
            category_probs={c: round(float(p), 4) for c, p in zip(self.categories, cat_probs)},
            sub_probs={c: round(float(p), 4) for c, p in zip(self.sub_categories, sub_probs)},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "backbone")
        self.tokenizer.save_pretrained(path / "tokenizer")
        torch.save(self.heads.state_dict(), path / "heads.pt")
        (path / "config.json").write_text(json.dumps({
            "backbone_name": self.backbone_name,
            "max_len": self.max_len,
            "categories": self.categories,
            "sub_categories": self.sub_categories,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("BERT 小模型已保存: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BertSmallModel":
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        categories = cfg.get("categories") or CATEGORIES
        sub_categories = cfg.get("sub_categories") or SUB_CATEGORIES
        backbone = AutoModel.from_pretrained(path / "backbone")
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        heads = _Heads(backbone.config.hidden_size, len(categories), len(sub_categories))
        heads.load_state_dict(torch.load(path / "heads.pt", map_location="cpu"))
        logger.info("BERT 小模型加载成功: %s (%d类/%d子类)", path, len(categories), len(sub_categories))
        return cls(backbone, tokenizer, heads, cfg.get("max_len", 64),
                   cfg.get("backbone_name", ""), categories=categories,
                   sub_categories=sub_categories)


def _from_pretrained(factory, name):
    """优先在线/缓存加载；网络不可用时强制走本地缓存，避免训练被 Hub 超时阻断。"""
    try:
        return factory.from_pretrained(name)
    except Exception as e:
        logger.warning("在线加载 %s 失败(%s)，改用本地缓存", name, e)
        return factory.from_pretrained(name, local_files_only=True)


def train_bert(
    texts: list[str],
    categories: list[str],
    sub_categories: list[str],
    output: str | Path,
    backbone_name: str = "hfl/rbt3",
    max_len: int = 64,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-5,
    test_size: float = 0.15,
    seed: int = 42,
    groups: list[str] | None = None,
) -> dict:
    """多任务训练 BERT 小模型，返回训练报告（含留出集 L1/L2 准确率）。

    groups 提供时按组切分（如模板 ID），避免同一模板同时出现在训练/测试集，
    使指标反映对未见表达的泛化能力。
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if len(texts) < 30:
        raise ValueError(f"训练样本过少: {len(texts)}")

    cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
    sub_idx = {s: i for i, s in enumerate(SUB_CATEGORIES)}
    y_cat = [cat_idx[c] for c in categories]
    y_sub = [sub_idx[s] for s in sub_categories]

    # 优先按组(模板)切分；无分组信息或分组不足以切分时退回分层切分
    indices = list(range(len(texts)))
    use_groups = bool(groups) and len(groups) == len(texts) and len(set(groups)) > 1
    if use_groups:
        try:
            tr_idx, te_idx = next(GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed,
            ).split(indices, groups=groups))
        except ValueError as e:
            logger.warning("按组切分失败(%s)，退回分层切分", e)
            use_groups = False
    if use_groups:
        split_mode = "group(template)"
    else:
        strat = [f"{c}/{s}" for c, s in zip(categories, sub_categories)]
        try:
            tr_idx, te_idx = train_test_split(
                indices, test_size=test_size, random_state=seed, stratify=strat,
            )
        except ValueError:
            tr_idx, te_idx = train_test_split(
                indices, test_size=test_size, random_state=seed,
            )
        split_mode = "random(stratified)"

    tokenizer = _from_pretrained(AutoTokenizer, backbone_name)
    backbone = _from_pretrained(AutoModel, backbone_name)
    model = BertSmallModel(
        backbone, tokenizer,
        _Heads(backbone.config.hidden_size, len(CATEGORIES), len(SUB_CATEGORIES)),
        max_len=max_len, backbone_name=backbone_name,
        categories=CATEGORIES, sub_categories=SUB_CATEGORIES,
    )

    def _encode(indices: list[int]):
        enc = tokenizer(
            [texts[i] for i in indices], truncation=True, max_length=max_len,
            padding="max_length", return_tensors="pt",
        )
        return TensorDataset(
            enc["input_ids"], enc["attention_mask"],
            torch.tensor([y_cat[i] for i in indices]),
            torch.tensor([y_sub[i] for i in indices]),
        )

    train_loader = DataLoader(_encode(tr_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(_encode(te_idx), batch_size=batch_size)

    params = list(model.backbone.parameters()) + list(model.heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    best_l2, best_epoch, best_l1 = -1.0, 0, 0.0

    for epoch in range(1, epochs + 1):
        model.backbone.train()
        model.heads.train()
        total_loss = 0.0
        for input_ids, attention_mask, c_batch, s_batch in train_loader:
            input_ids, attention_mask = input_ids.to(model.device), attention_mask.to(model.device)
            c_batch, s_batch = c_batch.to(model.device), s_batch.to(model.device)
            pooled = model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            pooled = model.heads.dropout(pooled)
            loss = ce(model.heads.category(pooled), c_batch) + ce(model.heads.sub(pooled), s_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.backbone.eval()
        model.heads.eval()
        l1 = l2 = n = 0
        with torch.no_grad():
            for input_ids, attention_mask, c_batch, s_batch in test_loader:
                pooled = model.backbone(
                    input_ids=input_ids, attention_mask=attention_mask,
                ).last_hidden_state[:, 0]
                pred_c = model.heads.category(pooled).argmax(dim=-1)
                pred_s = model.heads.sub(pooled).argmax(dim=-1)
                n += len(c_batch)
                l1 += int((pred_c == c_batch).sum())
                l2 += int((pred_s == s_batch).sum())
        acc1, acc2 = l1 / n, l2 / n
        logger.info("epoch %d/%d loss=%.4f L1=%.4f L2=%.4f", epoch, epochs,
                    total_loss / len(train_loader), acc1, acc2)
        if acc2 >= best_l2:
            best_l2, best_epoch, best_l1 = acc2, epoch, acc1
            model.save(output)

    return {
        "backbone": backbone_name,
        "split_mode": split_mode,
        "train": len(tr_idx),
        "test": len(te_idx),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "level1_accuracy": round(best_l1, 4),
        "level2_accuracy": round(best_l2, 4),
        "model": str(output),
    }
