"""生成小程序分享卡配图（500×400，微信标准 5:4）。

用法：cd frontend/weapp && python scripts/make-share-card.py
产物：src/assets/share-card.png

参数全部写死在此处，改文案或配色后重跑即可。脚本与产物一并提交。
设计依据：docs/superpowers/specs/2026-09-15-weapp-landing-share-design.md
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CANVAS_W, CANVAS_H = 500, 400
BG = '#f6f8fa'          # 小程序 --c-bg，与端内一致
TEXT_COLOR = '#1f2328'  # 小程序 --c-text
SLOGAN = '专利情报，一问即得'
FONT_SIZE = 28
LOGO_W = 200            # 品牌图缩放后宽度
GAP = 20                # 品牌图与标语间距

# 按平台候选，取第一个存在的。分享卡是离线一次性产物，不参与构建链路。
FONT_CANDIDATES = (
    'C:/Windows/Fonts/msyhbd.ttc',                              # 微软雅黑 Bold
    '/System/Library/Fonts/PingFang.ttc',                       #  macOS 苹方
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',      # Linux Noto CJK
)

ROOT = Path(__file__).resolve().parent.parent
SRC_LOGO = ROOT / 'src' / 'assets' / 'brand-logo.png'
OUT = ROOT / 'src' / 'assets' / 'share-card.png'


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit(f'找不到可用的中文字体，候选路径：{FONT_CANDIDATES}')


def main():
    logo = Image.open(SRC_LOGO).convert('RGBA')
    logo_h = round(LOGO_W * logo.height / logo.width)
    logo = logo.resize((LOGO_W, logo_h), Image.LANCZOS)

    font = load_font(FONT_SIZE)
    # 用 bbox 而非 font size 量文字高度：字体自带行距会让垂直居中偏下。
    bbox = font.getbbox(SLOGAN)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    top = (CANVAS_H - (logo_h + GAP + text_h)) // 2

    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), BG)
    # 第三个参数是 mask：品牌图有透明通道，必须走 alpha 合成，否则会糊成黑块。
    canvas.paste(logo, ((CANVAS_W - LOGO_W) // 2, top), logo)

    draw = ImageDraw.Draw(canvas)
    # 减去 bbox 左上角偏移，让文字按视觉外框居中，而不是按字体基线框。
    draw.text(
        ((CANVAS_W - text_w) // 2 - bbox[0], top + logo_h + GAP - bbox[1]),
        SLOGAN,
        font=font,
        fill=TEXT_COLOR,
    )

    canvas.save(OUT, optimize=True)
    print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {canvas.size[0]}x{canvas.size[1]})')


if __name__ == '__main__':
    main()
