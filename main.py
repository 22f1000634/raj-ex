import os, re, json, base64, tempfile, subprocess, sys, shutil, zipfile, tarfile
from io import BytesIO, StringIO
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import pdfplumber

# LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

load_dotenv()
app = FastAPI(title="Hybrid Data Analyst Agent")

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Gemini Fallback --------------------
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 6) if os.getenv(f"gemini_api_{i}")]
if not GEMINI_KEYS:
    raise RuntimeError("No Gemini API keys set in .env")

class LLMWithFallback:
    def __init__(self, models=None, temperature=0):
        self.keys = GEMINI_KEYS
        self.models = models or ["gemini-2.5-pro", "gemini-2.5-flash"]
        self.temperature = temperature

    def _get_llm_instance(self):
        for model in self.models:
            for key in self.keys:
                try:
                    return ChatGoogleGenerativeAI(model=model, temperature=self.temperature, google_api_key=key)
                except Exception:
                    continue
        raise RuntimeError("All Gemini keys/models failed")

    def bind_tools(self, tools):
        return self._get_llm_instance().bind_tools(tools)

    def invoke(self, prompt):
        return self._get_llm_instance().invoke(prompt)

# -------------------- Scraping Tool --------------------
@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """Scrape a URL (CSV, Excel, JSON, HTML table) and return dataframe as dict"""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        df = None

        if "csv" in ctype or url.endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif url.endswith((".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
        elif url.endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))
        elif "json" in ctype or url.endswith(".json"):
            data = resp.json()
            df = pd.json_normalize(data)
        elif "html" in ctype or "wiki" in url:
            tables = pd.read_html(StringIO(resp.text))
            if tables: df = tables[0]
        if df is None:
            df = pd.DataFrame({"text": [resp.text]})

        df.columns = df.columns.map(str)
        return {"status":"success","data":df.to_dict("records"),"columns":df.columns.tolist()}
    except Exception as e:
        return {"status":"error","message":str(e)}

# -------------------- Local File Parser --------------------
def parse_uploaded_file(file: UploadFile, tmpdir: str) -> Dict[str, Any]:
    try:
        filename = file.filename.lower()
        path = os.path.join(tmpdir, file.filename)
        with open(path, "wb") as f:
            f.write(file.file.read())

        if filename.endswith(".csv"):
            df = pd.read_csv(path)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(path)
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif filename.endswith(".json"):
            data = json.load(open(path,"r",encoding="utf-8"))
            df = pd.json_normalize(data)
        elif filename.endswith(".pdf"):
            dfs=[]
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    try:
                        table = page.extract_table()
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            dfs.append(df)
                    except: continue
            df = pd.concat(dfs) if dfs else pd.DataFrame()
        elif filename.endswith((".txt",".md")):
            text=open(path,"r",encoding="utf-8",errors="ignore").read()
            df=pd.DataFrame({"text":[text]})
        elif filename.endswith((".zip",".tar",".tar.gz",".tgz")):
            extract_dir=os.path.join(tmpdir,"extracted")
            os.makedirs(extract_dir,exist_ok=True)
            if filename.endswith(".zip"):
                with zipfile.ZipFile(path,"r") as z: z.extractall(extract_dir)
            else:
                with tarfile.open(path,"r:*") as t: t.extractall(extract_dir)
            # read first CSV/JSON if present
            for root,dirs,files in os.walk(extract_dir):
                for f in files:
                    if f.endswith(".csv"): 
                        with open(os.path.join(root,f),"rb") as ff:
                            return parse_uploaded_file(UploadFile(filename=f,file=ff),tmpdir)
            df=pd.DataFrame({"files":os.listdir(extract_dir)})
        else:
            df=pd.DataFrame({"filename":[filename]})

        return {"status":"success","filename":file.filename,"data":df.to_dict("records"),"columns":df.columns.tolist()}
    except Exception as e:
        return {"status":"error","message":str(e)}

# -------------------- Sandbox Executor --------------------
def inspect_plot_to_base64():
    return """def plot_to_base64(max_bytes=100000):
    buf=BytesIO(); plt.savefig(buf,format='png',bbox_inches='tight',dpi=100)
    buf.seek(0); img=buf.getvalue()
    if len(img)<=max_bytes: return base64.b64encode(img).decode('ascii')
    return base64.b64encode(img[:max_bytes]).decode('ascii')"""

def execute_code(code: str, injected_df: pd.DataFrame=None) -> Dict[str, Any]:
    preamble = [
        "import json, pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
        inspect_plot_to_base64()
    ]
    if injected_df is not None:
        tmp_pickle = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl").name
        injected_df.to_pickle(tmp_pickle)
        preamble.append(f"df = pd.read_pickle(r'''{tmp_pickle}''')")
        preamble.append("data = df.to_dict(orient='records')")

    script = "\n".join(preamble) + "\nresults={}\n" + code + "\nprint(json.dumps({'status':'success','answers':results},default=str))"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    tmp.write(script); tmp.flush(); tmp.close()
    try:
        out = subprocess.run([sys.executable, tmp.name], capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return {"status":"error","message":out.stderr.strip() or out.stdout.strip()}
        return json.loads(out.stdout.strip())
    except Exception as e:
        return {"status":"error","message":str(e)}
    finally:
        os.unlink(tmp.name)

# -------------------- Agent Setup --------------------
llm = LLMWithFallback(temperature=0)
tools=[scrape_url_to_dataframe]
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict Data Analyst Agent.
Return ONLY valid JSON in this structure:
{{
  "questions": ["Q1","Q2"],
  "code": "Python code that fills results = {{...}}"
}}
- Do not explain
- Do not add extra text
- If unsure, return {{"questions":[],"code":"results={{}}"}}"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------------------- API --------------------
@app.post("/api/")
async def analyze(questions: UploadFile = File(...), file: UploadFile = File(None)):
    try:
        tmpdir=tempfile.mkdtemp()
        q_text = (await questions.read()).decode("utf-8")

        df=None
        if file:
            parsed=parse_uploaded_file(file,tmpdir)
            if parsed.get("status")=="success":
                df=pd.DataFrame(parsed["data"])
                q_text+="\n\nDATA PREVIEW:\n"+str(df.head(3).to_dict())

        agent_out = agent_executor.invoke({"input": q_text})
        raw = agent_out.get("output") or ""

        # Extract JSON strictly
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return JSONResponse(content={"status":"error","message":"Agent did not return JSON","raw": raw})

        try:
            parsed = json.loads(match.group(0))
        except:
            return JSONResponse(content={"status":"error","message":"Invalid JSON","raw": raw})

        code = parsed.get("code","")
        res = execute_code(code, injected_df=df)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(content={"status":"error","message":str(e)})
    finally:
        shutil.rmtree(tmpdir,ignore_errors=True)
