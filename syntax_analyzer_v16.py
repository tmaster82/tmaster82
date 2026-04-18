import spacy
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import json

# 1. 모델 로드
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    print("모델 다운로드 중...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    clear_output()

# 2단어 복합 전치사
COMPOUND_PREPS_2 = [
    ("due", "to"), ("because", "of"), ("thanks", "to"),
    ("instead", "of"), ("according", "to"), ("as", "for"),
    ("apart", "from"), ("out", "of"), ("along", "with"),
    ("together", "with"), ("except", "for"), ("prior", "to"),
    ("regardless", "of"), ("contrary", "to"), ("ahead", "of"),
    ("as", "of"), ("up", "to"), ("next", "to"),
    ("close", "to"), ("owing", "to"),
]

# 3단어 복합 전치사
COMPOUND_PREPS_3 = [
    ("in", "spite", "of"), ("in", "front", "of"),
    ("as", "well", "as"), ("in", "addition", "to"),
    ("on", "account", "of"), ("by", "means", "of"),
    ("on", "behalf", "of"), ("in", "case", "of"),
    ("in", "place", "of"), ("in", "terms", "of"),
    ("with", "regard", "to"), ("with", "respect", "to"),
    ("in", "order", "to"), ("for", "fear", "of"),
    ("by", "way", "of"), ("in", "view", "of"),
]


def visual_syntax_analysis_v16(text):
    doc = nlp(text)
    tokens_data = []
    arrows = []

    # ---------------------------------------------------------
    # [0] 복합 전치사 감지 (3단어 → 2단어 순으로 greedy 매칭)
    # ---------------------------------------------------------
    compound_indices = set()
    compound_starts = set()

    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i].text.lower(), doc[i+1].text.lower(), doc[i+2].text.lower()
        for c1, c2, c3 in COMPOUND_PREPS_3:
            if t1 == c1 and t2 == c2 and t3 == c3:
                compound_starts.add(i)
                compound_indices.update([i, i+1, i+2])
                break

    for i in range(len(doc) - 1):
        if i in compound_indices:
            continue
        t1, t2 = doc[i].text.lower(), doc[i+1].text.lower()
        for c1, c2 in COMPOUND_PREPS_2:
            if t1 == c1 and t2 == c2:
                compound_starts.add(i)
                compound_indices.update([i, i+1])
                break

    # 데이터 초기화
    for i, token in enumerate(doc):
        tokens_data.append({
            "id": i,
            "text": token.text, "before": "", "after": "", "label": "",
            "style_class": "", "inline_style": ""
        })

    # ---------------------------------------------------------
    # [A] 가주어/진주어 & 가목적어/진목적어 (보라색 네모 박스)
    # ---------------------------------------------------------
    for i, token in enumerate(doc):
        if token.text.lower() == "it":

            if token.dep_ in ["nsubj", "expl"]:
                head_verb = token.head
                real_subject_start = None
                for child in head_verb.children:
                    if child.dep_ in ["csubj", "xcomp", "ccomp", "advcl"]:
                        for gc in child.children:
                            if gc.dep_ == "aux" and gc.text.lower() == "to":
                                real_subject_start = gc.i
                                break
                        if real_subject_start is None:
                            for gc in child.children:
                                if gc.dep_ == "mark" and gc.text.lower() == "that":
                                    real_subject_start = gc.i
                                    break
                if real_subject_start is not None:
                    tokens_data[i]["label"] = "<span style='color:purple'>가S</span>"
                    tokens_data[i]["style_class"] += " rect-purple "
                    tokens_data[real_subject_start]["label"] = "<span style='color:purple'>진S</span>"
                    tokens_data[real_subject_start]["style_class"] += " rect-purple "

            elif token.dep_ == "dobj":
                head_verb = token.head
                real_object_start = None
                for child in head_verb.children:
                    if child.dep_ in ["xcomp", "ccomp"]:
                        for gc in child.children:
                            if gc.dep_ == "aux" and gc.text.lower() == "to":
                                real_object_start = gc.i
                                break
                        if real_object_start is None:
                            for gc in child.children:
                                if gc.dep_ == "mark" and gc.text.lower() == "that":
                                    real_object_start = gc.i
                                    break
                if real_object_start is not None:
                    tokens_data[i]["label"] = "<span style='color:purple'>가O</span>"
                    tokens_data[i]["style_class"] += " rect-purple "
                    tokens_data[real_object_start]["label"] = "<span style='color:purple'>진O</span>"
                    tokens_data[real_object_start]["style_class"] += " rect-purple "

    # ---------------------------------------------------------
    # [B] 끊어 읽기 및 시각화
    # ---------------------------------------------------------
    for i, token in enumerate(doc):
        add_slash_before = False

        if i in compound_starts:
            add_slash_before = True
        elif i in compound_indices:
            add_slash_before = False
        elif token.pos_ in ["VERB", "AUX"]:
            if token.dep_ not in ["amod", "acl"]:
                if token.head.text.lower() == "of":
                    add_slash_before = False
                else:
                    if i == 0 or doc[i-1].pos_ not in ["VERB", "AUX", "ADV", "PART"]:
                        add_slash_before = True
                    is_verb_end = (i == len(doc)-1) or (doc[i+1].pos_ not in ["VERB", "AUX", "ADV", "PART"])
                    if is_verb_end and i < len(doc)-1:
                        next_is_compound = (i+1 in compound_starts)
                        if (doc[i+1].pos_ not in ["ADP", "SCONJ"] or doc[i+1].text.lower() == "of") or next_is_compound:
                            if doc[i+1].pos_ != "PUNCT":
                                tokens_data[i]["after"] += "<span class='slash'>/</span>"
        elif token.pos_ in ["ADP", "SCONJ"]:
            if token.text.lower() == "of":
                add_slash_before = False
            else:
                has_adv_modifier = (i > 0 and doc[i-1].pos_ == "ADV" and doc[i-1].head == token)
                if not has_adv_modifier:
                    add_slash_before = True
        elif token.pos_ == "ADV" and i + 1 < len(doc):
            if doc[i+1].pos_ in ["ADP", "SCONJ"] and token.head == doc[i+1]:
                add_slash_before = True

        if add_slash_before and i > 0:
            if doc[i-1].pos_ != "PUNCT":
                tokens_data[i]["before"] += "<span class='slash'>/</span>"

        # 관계절 괄호
        if token.dep_ == "relcl":
            left = token.left_edge.i
            right = token.right_edge.i
            tokens_data[left]["before"] = "<span class='bracket'>(</span>" + tokens_data[left]["before"]
            tokens_data[right]["after"] += "<span class='bracket'>)</span>"

        # 등위접속사 삼각형
        if token.text.lower() in ["and", "but", "or"]:
            tokens_data[i]["style_class"] += " triangle-outline "

        # 분사
        if token.tag_ == "VBG" and token.dep_ != "ROOT":
            if token.head.pos_ != "ADP":
                tokens_data[i]["style_class"] += " circle "
                if "span" not in tokens_data[i]["label"]:
                    tokens_data[i]["label"] = "현/분"
        elif token.tag_ == "VBN" and token.dep_ != "ROOT" and (i > 0 and doc[i-1].lemma_ != "have"):
            tokens_data[i]["style_class"] += " circle "
            if "span" not in tokens_data[i]["label"]:
                tokens_data[i]["label"] = "과/분"

        # 쉼표 + 관계사
        if token.text == "," and i + 1 < len(doc):
            if doc[i+1].tag_ in ["WP", "WDT", "WRB"]:
                tokens_data[i]["style_class"] += " circle-red "
                tokens_data[i+1]["inline_style"] += "text-decoration: underline; text-decoration-color: red; text-underline-offset: 3px;"

        # of 화살표
        if token.text.lower() == "of" and token.pos_ == "ADP" and i not in compound_indices:
            tokens_data[i]["style_class"] += " circle-red-tight "
            target_B = None
            for child in token.children:
                if child.dep_ in ["pobj", "pcomp"]:
                    target_B = child.i
                    break
            target_A = token.head.i
            if target_B is not None:
                arrows.append({"start": target_B, "end": target_A, "type": "top", "color": "red"})

        # 관계절 동사 → 수식 대상 화살표
        if token.dep_ == "relcl" and token.pos_ in ["VERB", "AUX"]:
            arrows.append({"start": i, "end": token.head.i, "type": "bottom", "color": "blue"})
            if not tokens_data[i]["label"]:
                tokens_data[i]["label"] = "V"

    # ---------------------------------------------------------
    # [C] 상관접속사 (확장: both/either/neither/not only)
    # ---------------------------------------------------------
    def _mark_correlative(idx_a, idx_b, circle_indices):
        for ci in circle_indices:
            tokens_data[ci]["style_class"] += " circle-red "
        a = idx_a
        if a < len(doc) and doc[a].pos_ == "DET" and a + 1 < len(doc):
            a += 1
        if a < len(doc):
            tokens_data[a]["label"] = "A"
        b = idx_b
        if b < len(doc) and doc[b].pos_ == "DET" and b + 1 < len(doc):
            b += 1
        if b < len(doc):
            tokens_data[b]["label"] = "B"

    for i, token in enumerate(doc):
        w = token.text.lower()

        # not...but  /  not only...but (also)
        if w == "not":
            # not only...but also
            if i + 1 < len(doc) and doc[i+1].text.lower() == "only":
                for j in range(i + 2, min(i + 25, len(doc))):
                    if doc[j].text.lower() == "but":
                        circle = [i, i+1, j]
                        if j + 1 < len(doc) and doc[j+1].text.lower() == "also":
                            circle.append(j+1)
                        a_start = i + 2
                        b_start = j + (2 if j+1 < len(doc) and doc[j+1].text.lower() == "also" else 1)
                        _mark_correlative(a_start, b_start, circle)
                        break
            else:
                # not...but
                for j in range(i + 1, min(i + 15, len(doc))):
                    if doc[j].text.lower() == "but" and doc[j].dep_ != "prep":
                        _mark_correlative(i + 1, j + 1, [i, j])
                        break

        # both...and
        elif w == "both":
            for j in range(i + 1, min(i + 20, len(doc))):
                if doc[j].text.lower() == "and":
                    _mark_correlative(i + 1, j + 1, [i, j])
                    break

        # either...or
        elif w == "either":
            for j in range(i + 1, min(i + 20, len(doc))):
                if doc[j].text.lower() == "or":
                    _mark_correlative(i + 1, j + 1, [i, j])
                    break

        # neither...nor
        elif w == "neither":
            for j in range(i + 1, min(i + 20, len(doc))):
                if doc[j].text.lower() == "nor":
                    _mark_correlative(i + 1, j + 1, [i, j])
                    break

        # whether...or
        elif w == "whether":
            for j in range(i + 1, min(i + 20, len(doc))):
                if doc[j].text.lower() == "or":
                    _mark_correlative(i + 1, j + 1, [i, j])
                    break

    # ---------------------------------------------------------
    # [D] 주요 문장 성분 레이블 S / V / O / C / IO
    # (가주어·분사·상관접속사 레이블이 없는 토큰에만 적용)
    # ---------------------------------------------------------
    def _not_labeled(idx):
        lbl = tokens_data[idx]["label"]
        return lbl == "" or lbl in ["V"]  # relcl V는 덮어써도 무방

    SVOC_STYLE = "border-bottom: 2.5px solid #1565C0; padding-bottom: 1px;"

    for token in doc:
        i = token.i
        if not _not_labeled(i):
            continue

        if token.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>V</span>"
            tokens_data[i]["inline_style"] += SVOC_STYLE

        elif token.dep_ in ["nsubj", "nsubjpass"] and token.head.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>S</span>"
            tokens_data[i]["inline_style"] += SVOC_STYLE

        elif token.dep_ == "dobj" and token.head.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>O</span>"
            tokens_data[i]["inline_style"] += SVOC_STYLE

        elif token.dep_ in ["attr", "acomp"] and token.head.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>C</span>"
            tokens_data[i]["inline_style"] += SVOC_STYLE

        elif token.dep_ in ["dative", "iobj"] and token.head.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>IO</span>"
            tokens_data[i]["inline_style"] += SVOC_STYLE

    # ---------------------------------------------------------
    # HTML 조립
    # ---------------------------------------------------------
    arrows_json = json.dumps(arrows)
    html_content = ""

    for t in tokens_data:
        if "\n" in t['text']:
            html_content += "<div style='flex-basis: 100%; height: 50px;'></div>"
            continue

        if t['before']:
            html_content += t['before']

        display_label = t['label'] if t['label'] else "&nbsp;"
        html_content += f"""
        <div class='word-unit' id='unit-{t['id']}'>
            <span class='word {t['style_class']}' style='{t['inline_style']}' id='token-{t['id']}'>{t['text']}</span>
            <span class='meta-label'>{display_label}</span>
        </div>
        """
        if t['after']:
            html_content += t['after']

    legend_html = """
    <div style="font-family:'Segoe UI',sans-serif;font-size:12px;margin-top:18px;padding:10px 14px;
                border:1px solid #ddd;border-radius:8px;background:#f9f9f9;
                display:flex;flex-wrap:wrap;gap:12px;align-items:center;">
        <strong style="width:100%;margin-bottom:4px;color:#333;">범례 (Legend)</strong>
        <span><u style="text-decoration-color:#1565C0;text-decoration-thickness:2.5px;">S/V/O/C</u> 주요 문장 성분</span>
        <span><span style="border:2px solid purple;border-radius:4px;padding:0 3px;font-size:11px;">가S/진S</span> 가·진주어</span>
        <span><span style="border:2px solid #ffcc00;border-radius:50%;padding:0 3px;font-size:11px;">현/분</span> 현재분사</span>
        <span><span style="border:2px solid #ffcc00;border-radius:50%;padding:0 3px;font-size:11px;">과/분</span> 과거분사</span>
        <span><span style="border:2px solid red;border-radius:50%;padding:0 3px;font-size:11px;">both…and</span> 상관접속사 A/B</span>
        <span><span style="color:green;font-weight:bold;">( )</span> 관계절</span>
        <span><span style="color:blue;font-weight:bold;">/</span> 끊어읽기</span>
        <span><span style="color:orange;">△</span> 등위접속사</span>
        <span>빨간↑ of 수식 방향 &nbsp;|&nbsp; 파란↓ 관계절 수식</span>
    </div>
    """

    full_html = f"""
    <div id="syntax-container" class="sentence-box" style="position:relative;">
        <svg id="arrow-layer" style="position:absolute;top:0;left:0;width:100%;height:100%;
             pointer-events:none;z-index:0;overflow:visible;"></svg>
        {html_content}
    </div>
    {legend_html}

    <style>
        .sentence-box {{
            font-family: 'Segoe UI', sans-serif; font-size: 20px;
            line-height: 2.2; display: flex; flex-wrap: wrap;
            align-items: flex-end; padding: 20px 10px;
        }}
        .word-unit {{
            display: inline-flex; flex-direction: column; align-items: center;
            margin: 0 6px; vertical-align: bottom; position: relative; z-index: 1;
        }}
        .word {{ padding: 2px 4px; position: relative; }}
        .meta-label {{
            font-size: 11px; color: #555; font-weight: bold;
            margin-top: 4px; min-height: 18px; display: block;
        }}
        .slash {{
            color: blue; font-weight: 900; margin: 0 4px;
            font-size: 1.1em; align-self: center; margin-bottom: 22px;
        }}
        .bracket {{
            color: green; font-weight: 900; font-size: 1.2em;
            align-self: center; margin-bottom: 22px;
        }}
        .triangle-outline {{
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><polygon points="50,15 90,85 10,85" fill="none" stroke="orange" stroke-width="8"/></svg>');
            background-repeat: no-repeat; background-position: center; background-size: 100% 100%;
        }}
        .circle        {{ border: 2px solid #ffcc00; border-radius: 50%; }}
        .circle-red    {{ border: 2px solid red;     border-radius: 50%; }}
        .circle-red-tight {{ border: 2px solid red; border-radius: 50%; padding: 0px 2px; }}
        .rect-purple   {{ border: 2px solid #800080; border-radius: 4px; padding: 0px 3px; }}
    </style>

    <script>
    (function() {{
        const arrows = {arrows_json};

        function drawArrows() {{
            const container = document.getElementById('syntax-container');
            const svg       = document.getElementById('arrow-layer');
            if (!container || !svg) return;

            svg.innerHTML = '';
            const cRect = container.getBoundingClientRect();

            const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
            ['red', 'blue'].forEach(color => {{
                const marker  = document.createElementNS("http://www.w3.org/2000/svg", "marker");
                marker.setAttribute("id",          `arrowhead-${{color}}`);
                marker.setAttribute("markerWidth",  "10");
                marker.setAttribute("markerHeight", "7");
                marker.setAttribute("refX",         "9");
                marker.setAttribute("refY",         "3.5");
                marker.setAttribute("orient",       "auto");
                const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
                poly.setAttribute("points", "0 0, 10 3.5, 0 7");
                poly.setAttribute("fill",   color);
                marker.appendChild(poly);
                defs.appendChild(marker);
            }});
            svg.appendChild(defs);

            arrows.forEach(arrow => {{
                const sEl = document.getElementById('token-' + arrow.start);
                const eEl = document.getElementById('token-' + arrow.end);
                if (!sEl || !eEl) return;

                const sR = sEl.getBoundingClientRect();
                const eR = eEl.getBoundingClientRect();
                const sx = sR.left + sR.width / 2 - cRect.left;
                const ex = eR.left + eR.width / 2 - cRect.left;

                let pathData;
                if (arrow.type === 'top') {{
                    const sy    = sR.top  - cRect.top;
                    const ey    = eR.top  - cRect.top;
                    const pivot = Math.min(sy, ey) - 15;
                    pathData = `M ${{sx}} ${{sy}} L ${{sx}} ${{pivot}} L ${{ex}} ${{pivot}} L ${{ex}} ${{ey}}`;
                }} else {{
                    const sy    = sR.bottom - cRect.bottom + 15;
                    const ey    = eR.bottom - cRect.bottom + 15;
                    const pivot = Math.max(sy, ey) + 15;
                    pathData = `M ${{sx}} ${{sy}} L ${{sx}} ${{pivot}} L ${{ex}} ${{pivot}} L ${{ex}} ${{ey}}`;
                }}

                const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                path.setAttribute("d",           pathData);
                path.setAttribute("fill",        "none");
                path.setAttribute("stroke",      arrow.color);
                path.setAttribute("stroke-width","1.5");
                path.setAttribute("marker-end",  `url(#arrowhead-${{arrow.color}})`);
                svg.appendChild(path);
            }});
        }}

        setTimeout(drawArrows, 800);
        window.addEventListener('resize', drawArrows);
    }})();
    </script>
    """
    return full_html


# =============================================================
# UI
# =============================================================
input_area = widgets.Textarea(
    value='',
    placeholder=(
        "테스트 문장 (한 줄에 하나씩 입력):\n"
        "It is dangerous to swim here.\n"
        "I found it hard to believe his story.\n"
        "Not only cats but also dogs are allowed.\n"
        "Both students and teachers must wear masks.\n"
        "Either you or I am wrong.\n"
        "Neither rain nor snow stopped us.\n"
        "She succeeded in spite of many difficulties.\n"
        "The book on the shelf belongs to me."
    ),
    layout=widgets.Layout(width='100%', height='180px')
)

analyze_button = widgets.Button(
    description='v16 분석 시작',
    button_style='primary',
    icon='search',
    layout=widgets.Layout(width='160px')
)

output_area = widgets.Output()


def on_analyze_click(b):
    with output_area:
        clear_output()
        text = input_area.value
        if not text.strip():
            return
        print("▼ 분석 결과 (v16):")
        try:
            display(HTML(visual_syntax_analysis_v16(text)))
        except Exception as e:
            print(f"오류: {e}")


analyze_button.on_click(on_analyze_click)

display(widgets.HTML(
    "<h3>📐 파이프라인 구문 분석기 "
    "(v16: S/V/O/C 레이블 · 상관접속사 확장 · 복합전치사 확장 · 범례)</h3>"
))
display(input_area)
display(analyze_button)
display(output_area)
