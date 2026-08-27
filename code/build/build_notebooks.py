# -*- coding: utf-8 -*-
"""노트북 빌더 — nbformat 없이 .ipynb(v4) JSON 을 직접 만들고, **출력까지 저장**한다.

왜 출력을 저장하나. 이 논문의 원자료(NPS 사업장 등록부·PitchBook·상용 재무DB·KRX 주가)는 라이선스라
저장소에 올릴 수 없다. 따라서 방문자가 노트북을 실행할 수는 없다. 대신 **집계 산출물(JSON)만으로 표·그림을
전부 재생성**하도록 짜고, 실행 결과를 노트북에 담아 GitHub 에서 코드와 결과를 같이 보게 한다.
"""
import base64,io,json,os,sys,contextlib
def build(path,title,intro,cells):
    # 노트북이 놓일 디렉터리에서 실행한다 — 상대경로(../artifacts)가 사용자 실행 환경과 같아야 한다
    here=os.getcwd(); os.chdir(os.path.dirname(os.path.abspath(path)) or ".")
    ns={}; out_cells=[{"cell_type":"markdown","metadata":{},"source":[title+"\n","\n"]+[l+"\n" for l in intro]}]
    for md,code in cells:
        if md: out_cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in md]})
        buf=io.StringIO(); outputs=[]
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        with contextlib.redirect_stdout(buf):
            exec(code,ns)
        txt=buf.getvalue()
        # rich outputs handed back by the cell: `_md` (markdown string) and `_figs` (PNG bytes)
        if ns.get("_md"):
            outputs.append({"output_type":"display_data","metadata":{},
                            "data":{"text/markdown":[l+"\n" for l in ns["_md"].split("\n")],
                                    "text/plain":[l+"\n" for l in ns["_md"].split("\n")]}}); ns["_md"]=None
        for png in ns.get("_figs",[]) or []:
            outputs.append({"output_type":"display_data","metadata":{},
                            "data":{"image/png":base64.b64encode(png).decode()}})
        if "_figs" in ns: ns["_figs"].clear()
        if txt.strip():
            outputs.append({"output_type":"stream","name":"stdout","text":[l+"\n" for l in txt.rstrip("\n").split("\n")]})
        for num in plt.get_fignums():
            fig=plt.figure(num); b=io.BytesIO(); fig.savefig(b,format="png",dpi=140,bbox_inches="tight")
            outputs.append({"output_type":"display_data","metadata":{},
                            "data":{"image/png":base64.b64encode(b.getvalue()).decode()}})
            plt.close(fig)
        out_cells.append({"cell_type":"code","execution_count":None,"metadata":{},
                          "source":[l+"\n" for l in code.rstrip("\n").split("\n")],"outputs":outputs})
    nb={"cells":out_cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":sys.version.split()[0]}},"nbformat":4,"nbformat_minor":5}
    os.chdir(here)
    json.dump(nb,open(path,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
    print(f"  {path}  ({len(out_cells)} cells)")
