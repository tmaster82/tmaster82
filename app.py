import spacy
import streamlit as st
import streamlit.components.v1 as components
import json
import uuid

# ── 모델 로드 ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_model()

# ── 복합 전치사 목록 ──────────────────────────────────────────
COMPOUND_PREPS_2 = [
    ("due", "to"), ("because", "of"), ("thanks", "to"),
    ("instead", "of"), ("according", "to"), ("as", "for"),
    ("apart", "from"), ("out", "of"), ("along", "with"),
    ("together", "with"), ("except", "for"), ("prior", "to"),
    ("regardless", "of"), ("contrary", "to"), ("ahead", "of"),
    ("as", "of"), ("up", "to"), ("next", "to"),
    ("close", "to"), ("owing", "to"),
]
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


# ── 핵심 분석 함수 ────────────────────────────────────────────
def analyze(text: str) -> str:
    """문장을 분석해 HTML 문자열을 반환합니다."""
    uid = uuid.uuid4().hex[:8]
    doc = nlp(text)
    tokens_data = []
    arrows = []

    # [0] 복합 전치사 감지
    compound_indices: set = set()
    compound_starts: set = set()

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

    for i, token in enumerate(doc):
        tokens_data.append({
            "id": i, "text": token.text,
            "before": "", "after": "", "label": "",
            "style_class": "", "inline_style": ""
        })

    # [A] 가주어/진주어 & 가목적어/진목적어
    for i, token in enumerate(doc):
        if token.text.lower() != "it":
            continue
        if token.dep_ in ["nsubj", "expl"]:
            real = None
            for child in token.head.children:
                if child.dep_ in ["csubj", "xcomp", "ccomp", "advcl"]:
                    for gc in child.children:
                        if gc.dep_ == "aux" and gc.text.lower() == "to":
                            real = gc.i; break
                    if real is None:
                        for gc in child.children:
                            if gc.dep_ == "mark" and gc.text.lower() == "that":
                                real = gc.i; break
            if real is not None:
                tokens_data[i]["label"] = "<span style='color:purple'>가S</span>"
                tokens_data[i]["style_class"] += " rect-purple"
                tokens_data[real]["label"] = "<span style='color:purple'>진S</span>"
                tokens_data[real]["style_class"] += " rect-purple"
        elif token.dep_ == "dobj":
            real = None
            for child in token.head.children:
                if child.dep_ in ["xcomp", "ccomp"]:
                    for gc in child.children:
                        if gc.dep_ == "aux" and gc.text.lower() == "to":
                            real = gc.i; break
                    if real is None:
                        for gc in child.children:
                            if gc.dep_ == "mark" and gc.text.lower() == "that":
                                real = gc.i; break
            if real is not None:
                tokens_data[i]["label"] = "<span style='color:purple'>가O</span>"
                tokens_data[i]["style_class"] += " rect-purple"
                tokens_data[real]["label"] = "<span style='color:purple'>진O</span>"
                tokens_data[real]["style_class"] += " rect-purple"

    # [B] 끊어읽기 / 도형 / 화살표
    for i, token in enumerate(doc):
        slash_before = False

        if i in compound_starts:
            slash_before = True
        elif i in compound_indices:
            slash_before = False
        elif token.pos_ in ["VERB", "AUX"]:
            if token.dep_ not in ["amod", "acl"]:
                if token.head.text.lower() != "of":
                    if i == 0 or doc[i-1].pos_ not in ["VERB", "AUX", "ADV", "PART"]:
                        slash_before = True
                    is_end = (i == len(doc)-1) or doc[i+1].pos_ not in ["VERB", "AUX", "ADV", "PART"]
                    if is_end and i < len(doc)-1:
                        nxt_compound = (i+1 in compound_starts)
                        if (doc[i+1].pos_ not in ["ADP", "SCONJ"] or doc[i+1].text.lower() == "of") or nxt_compound:
                            if doc[i+1].pos_ != "PUNCT":
                                tokens_data[i]["after"] += "<span class='slash'>/</span>"
        elif token.pos_ in ["ADP", "SCONJ"]:
            if token.text.lower() != "of":
                has_adv = i > 0 and doc[i-1].pos_ == "ADV" and doc[i-1].head == token
                if not has_adv:
                    slash_before = True
        elif token.pos_ == "ADV" and i+1 < len(doc):
            if doc[i+1].pos_ in ["ADP", "SCONJ"] and token.head == doc[i+1]:
                slash_before = True

        if slash_before and i > 0 and doc[i-1].pos_ != "PUNCT":
            tokens_data[i]["before"] += "<span class='slash'>/</span>"

        if token.dep_ == "relcl":
            l, r = token.left_edge.i, token.right_edge.i
            tokens_data[l]["before"] = "<span class='bracket'>(</span>" + tokens_data[l]["before"]
            tokens_data[r]["after"] += "<span class='bracket'>)</span>"

        if token.text.lower() in ["and", "but", "or"]:
            tokens_data[i]["style_class"] += " triangle-outline"

        if token.tag_ == "VBG" and token.dep_ != "ROOT" and token.head.pos_ != "ADP":
            tokens_data[i]["style_class"] += " circle"
            if "span" not in tokens_data[i]["label"]:
                tokens_data[i]["label"] = "현/분"
        elif token.tag_ == "VBN" and token.dep_ != "ROOT" and (i > 0 and doc[i-1].lemma_ != "have"):
            tokens_data[i]["style_class"] += " circle"
            if "span" not in tokens_data[i]["label"]:
                tokens_data[i]["label"] = "과/분"

        if token.text == "," and i+1 < len(doc) and doc[i+1].tag_ in ["WP", "WDT", "WRB"]:
            tokens_data[i]["style_class"] += " circle-red"
            tokens_data[i+1]["inline_style"] += "text-decoration:underline;text-decoration-color:red;text-underline-offset:3px;"

        if token.text.lower() == "of" and token.pos_ == "ADP" and i not in compound_indices:
            tokens_data[i]["style_class"] += " circle-red-tight"
            target_b = next((c.i for c in token.children if c.dep_ in ["pobj", "pcomp"]), None)
            if target_b is not None:
                arrows.append({"start": target_b, "end": token.head.i, "type": "top", "color": "red"})

        if token.dep_ == "relcl" and token.pos_ in ["VERB", "AUX"]:
            arrows.append({"start": i, "end": token.head.i, "type": "bottom", "color": "blue"})
            if not tokens_data[i]["label"]:
                tokens_data[i]["label"] = "V"

    # [C] 상관접속사
    def _mark(a_start, b_start, circle_idxs):
        for ci in circle_idxs:
            tokens_data[ci]["style_class"] += " circle-red"
        a = a_start
        if a < len(doc) and doc[a].pos_ == "DET" and a+1 < len(doc): a += 1
        if a < len(doc): tokens_data[a]["label"] = "A"
        b = b_start
        if b < len(doc) and doc[b].pos_ == "DET" and b+1 < len(doc): b += 1
        if b < len(doc): tokens_data[b]["label"] = "B"

    for i, token in enumerate(doc):
        w = token.text.lower()
        if w == "not":
            if i+1 < len(doc) and doc[i+1].text.lower() == "only":
                for j in range(i+2, min(i+25, len(doc))):
                    if doc[j].text.lower() == "but":
                        ci = [i, i+1, j]
                        also = j+1 < len(doc) and doc[j+1].text.lower() == "also"
                        if also: ci.append(j+1)
                        _mark(i+2, j+(2 if also else 1), ci); break
            else:
                for j in range(i+1, min(i+15, len(doc))):
                    if doc[j].text.lower() == "but" and doc[j].dep_ != "prep":
                        _mark(i+1, j+1, [i, j]); break
        elif w == "both":
            for j in range(i+1, min(i+20, len(doc))):
                if doc[j].text.lower() == "and": _mark(i+1, j+1, [i, j]); break
        elif w == "either":
            for j in range(i+1, min(i+20, len(doc))):
                if doc[j].text.lower() == "or": _mark(i+1, j+1, [i, j]); break
        elif w == "neither":
            for j in range(i+1, min(i+20, len(doc))):
                if doc[j].text.lower() == "nor": _mark(i+1, j+1, [i, j]); break
        elif w == "whether":
            for j in range(i+1, min(i+20, len(doc))):
                if doc[j].text.lower() == "or": _mark(i+1, j+1, [i, j]); break

    # [D] S/V/O/C/IO 주요 문장 성분
    UL = "border-bottom:2.5px solid #1565C0;padding-bottom:1px;"

    def _unlabeled(idx):
        lbl = tokens_data[idx]["label"]
        return lbl == "" or lbl == "V"

    for token in doc:
        i = token.i
        if not _unlabeled(i): continue
        if token.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>V</span>"
            tokens_data[i]["inline_style"] += UL
        elif token.dep_ in ["nsubj", "nsubjpass"] and token.head.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>S</span>"
            tokens_data[i]["inline_style"] += UL
        elif token.dep_ == "dobj" and token.head.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>O</span>"
            tokens_data[i]["inline_style"] += UL
        elif token.dep_ in ["attr", "acomp"] and token.head.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>C</span>"
            tokens_data[i]["inline_style"] += UL
        elif token.dep_ in ["dative", "iobj"] and token.head.dep_ == "ROOT":
            tokens_data[i]["label"] = "<span style='color:#1565C0;font-weight:bold'>IO</span>"
            tokens_data[i]["inline_style"] += UL

    # ── HTML 조립 ─────────────────────────────────────────────
    cid = f"sc-{uid}"   # syntax-container id
    aid = f"al-{uid}"   # arrow-layer id

    body = ""
    for t in tokens_data:
        if t["before"]: body += t["before"]
        lbl = t["label"] or "&nbsp;"
        body += (
            f"<div class='wu' id='u{uid}-{t['id']}'>"
            f"<span class='w {t['style_class']}' style='{t['inline_style']}'"
            f" id='t{uid}-{t['id']}'>{t['text']}</span>"
            f"<span class='ml'>{lbl}</span></div>"
        )
        if t["after"]: body += t["after"]

    arrows_json = json.dumps(arrows)

    legend = """
    <div style="font-family:'Segoe UI',sans-serif;font-size:12px;margin-top:14px;
                padding:10px 14px;border:1px solid #ddd;border-radius:8px;
                background:#f9f9f9;display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
      <strong style="width:100%;margin-bottom:2px;color:#333;">범례</strong>
      <span><u style="text-decoration-color:#1565C0;text-decoration-thickness:2.5px;">S/V/O/C</u> 주요 문장 성분</span>
      <span><span style="border:2px solid purple;border-radius:4px;padding:0 3px;font-size:11px;">가S/진S</span> 가·진주어</span>
      <span><span style="border:2px solid #ffcc00;border-radius:50%;padding:0 3px;font-size:11px;">현/분</span> 현재분사</span>
      <span><span style="border:2px solid #ffcc00;border-radius:50%;padding:0 3px;font-size:11px;">과/분</span> 과거분사</span>
      <span><span style="border:2px solid red;border-radius:50%;padding:0 3px;font-size:11px;">상관접속사</span> A/B 병렬</span>
      <span><span style="color:green;font-weight:bold;">( )</span> 관계절</span>
      <span><span style="color:blue;font-weight:bold;">/</span> 끊어읽기</span>
      <span><span style="color:orange;">△</span> 등위접속사</span>
      <span>빨간↑ of 수식 &nbsp;|&nbsp; 파란↓ 관계절 수식</span>
    </div>"""

    return f"""
    <div id="{cid}" style="position:relative;font-family:'Segoe UI',sans-serif;
         font-size:20px;line-height:2.4;display:flex;flex-wrap:wrap;
         align-items:flex-end;padding:16px 10px;">
      <svg id="{aid}" style="position:absolute;top:0;left:0;width:100%;height:100%;
           pointer-events:none;z-index:0;overflow:visible;"></svg>
      {body}
    </div>
    {legend}
    <style>
      .wu{{display:inline-flex;flex-direction:column;align-items:center;
           margin:0 5px;vertical-align:bottom;position:relative;z-index:1;}}
      .w{{padding:2px 4px;}}
      .ml{{font-size:11px;color:#555;font-weight:bold;margin-top:3px;min-height:16px;display:block;}}
      .slash{{color:blue;font-weight:900;margin:0 3px;font-size:1.1em;align-self:center;margin-bottom:20px;}}
      .bracket{{color:green;font-weight:900;font-size:1.2em;align-self:center;margin-bottom:20px;}}
      .triangle-outline{{background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><polygon points="50,15 90,85 10,85" fill="none" stroke="orange" stroke-width="8"/></svg>');background-repeat:no-repeat;background-position:center;background-size:100% 100%;}}
      .circle{{border:2px solid #ffcc00;border-radius:50%;}}
      .circle-red{{border:2px solid red;border-radius:50%;}}
      .circle-red-tight{{border:2px solid red;border-radius:50%;padding:0 2px;}}
      .rect-purple{{border:2px solid #800080;border-radius:4px;padding:0 3px;}}
    </style>
    <script>
    (function(){{
      const arrows={arrows_json};
      const uid="{uid}";
      function draw(){{
        const c=document.getElementById("{cid}");
        const s=document.getElementById("{aid}");
        if(!c||!s) return;
        s.innerHTML="";
        const cr=c.getBoundingClientRect();
        const defs=document.createElementNS("http://www.w3.org/2000/svg","defs");
        ["red","blue"].forEach(col=>{{
          const m=document.createElementNS("http://www.w3.org/2000/svg","marker");
          m.setAttribute("id","ah-"+col+"-{uid}");
          m.setAttribute("markerWidth","10");m.setAttribute("markerHeight","7");
          m.setAttribute("refX","9");m.setAttribute("refY","3.5");m.setAttribute("orient","auto");
          const p=document.createElementNS("http://www.w3.org/2000/svg","polygon");
          p.setAttribute("points","0 0,10 3.5,0 7");p.setAttribute("fill",col);
          m.appendChild(p);defs.appendChild(m);
        }});
        s.appendChild(defs);
        arrows.forEach(a=>{{
          const se=document.getElementById("t"+uid+"-"+a.start);
          const ee=document.getElementById("t"+uid+"-"+a.end);
          if(!se||!ee) return;
          const sr=se.getBoundingClientRect(),er=ee.getBoundingClientRect();
          const sx=sr.left+sr.width/2-cr.left,ex=er.left+er.width/2-cr.left;
          let d;
          if(a.type==="top"){{
            const sy=sr.top-cr.top,ey=er.top-cr.top,pv=Math.min(sy,ey)-15;
            d=`M ${{sx}} ${{sy}} L ${{sx}} ${{pv}} L ${{ex}} ${{pv}} L ${{ex}} ${{ey}}`;
          }}else{{
            const sy=sr.bottom-cr.bottom+15,ey=er.bottom-cr.bottom+15,pv=Math.max(sy,ey)+15;
            d=`M ${{sx}} ${{sy}} L ${{sx}} ${{pv}} L ${{ex}} ${{pv}} L ${{ex}} ${{ey}}`;
          }}
          const path=document.createElementNS("http://www.w3.org/2000/svg","path");
          path.setAttribute("d",d);path.setAttribute("fill","none");
          path.setAttribute("stroke",a.color);path.setAttribute("stroke-width","1.5");
          path.setAttribute("marker-end","url(#ah-"+a.color+"-{uid})");
          s.appendChild(path);
        }});
      }}
      setTimeout(draw,600);
      window.addEventListener("resize",draw);
    }})();
    </script>
    """


# ── Streamlit UI ──────────────────────────────────────────────
st.set_page_config(
    page_title="영어 구문 분석기",
    page_icon="📐",
    layout="wide",
)

st.title("📐 영어 구문 분석기")
st.caption("S/V/O/C · 가주어/진주어 · 분사구 · 상관접속사 · 복합전치사 · 관계절 시각화")

EXAMPLES = [
    "It is dangerous to swim here.",
    "I found it hard to believe his story.",
    "It is clear that he loves her.",
    "Not only cats but also dogs are allowed.",
    "Both students and teachers must wear masks.",
    "She succeeded in spite of many difficulties.",
    "The man who lives next door is a doctor.",
    "The discovery of the new planet excited scientists.",
]

with st.expander("예시 문장 보기", expanded=False):
    for ex in EXAMPLES:
        if st.button(ex, key=ex):
            st.session_state["input_text"] = ex

with st.form("form"):
    text_input = st.text_area(
        "분석할 문장을 입력하세요 (여러 문장은 줄바꿈으로 구분)",
        value=st.session_state.get("input_text", ""),
        height=130,
        placeholder="예: It is dangerous to swim here.",
    )
    submitted = st.form_submit_button("🔍 분석 시작", type="primary", use_container_width=True)

if submitted:
    sentences = [s.strip() for s in text_input.splitlines() if s.strip()]
    if not sentences:
        st.warning("문장을 입력해주세요.")
    else:
        for i, sent in enumerate(sentences):
            st.markdown(f"**{i+1}. {sent}**")
            html_out = analyze(sent)
            components.html(html_out, height=420, scrolling=True)
            if i < len(sentences) - 1:
                st.divider()
